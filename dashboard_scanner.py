"""Fingerprint dashboard-owned catalogue rows; never call Meta from Render."""
from urllib.parse import urlencode
import random
import time
import social_inventory as core
from scanner_quota import QuotaHTTP, scope

def bridge_client():
    client = core.Dashboard(core.SafeHTTP(core.Budget(seconds=60, max_bytes=8*1024*1024)))
    client.http = QuotaHTTP(client.http, scope(client.base, client.headers))
    return client

def read_bridge(client, path, params=None):
    url = client.base + '/api/ingest/scanner/' + path
    if params: url += '?' + urlencode(params)
    return client.http.json(url, client.hosts, headers=client.headers, limit=4*1024*1024)

def catalogue_accounts():
    data = read_bridge(bridge_client(), 'accounts')
    if data.get('source') != 'dashboard_catalogue' or not isinstance(data.get('rows'), list):
        raise core.SafeError('dashboard_catalogue_invalid')
    return data

def connection_diagnostics():
    try:
        data = read_bridge(bridge_client(), 'health')
        if data.get('ok') is not True or data.get('source') != 'dashboard_catalogue':
            raise core.SafeError('dashboard_bridge_response_invalid')
        return {'source':'dashboard_catalogue', 'dashboard':{'reachable':True, 'auth_verified':True},
                'meta_requests_from_render':False, 'cdn_access_verified':False}
    except core.SafeError as exc:
        return {'source':'dashboard_catalogue', 'dashboard':{'reachable':False, 'auth_verified':False,
                'code':str(exc), **getattr(exc, 'diagnostic', {})}, 'cdn_access_verified':False}
    except Exception:
        return {'source':'dashboard_catalogue', 'dashboard':{'reachable':False, 'auth_verified':False,
                'code':'diagnostic_failed'}, 'cdn_access_verified':False}

class FingerprintSink(core.Dashboard):
    def delivery_http(self):
        return QuotaHTTP(core.SafeHTTP(core.Budget(seconds=30, max_bytes=8*1024*1024)), scope(self.base, self.headers))

    def write_batch(self, endpoint, payload):
        # Only deterministic fingerprint updates and group upserts use this retry path.
        transient = {'store_rate_limited', 'store_network_unavailable', 'store_google_unavailable',
                     'store_read_conflict_retry', 'store_revision_conflict', 'store_read_incomplete',
                     'store_response_not_json'}
        for attempt in range(3):
            try:
                return self.http.json(self.base+endpoint,self.hosts,method='POST',headers=self.headers,payload=payload)
            except core.SafeError as exc:
                diagnostic = dict(getattr(exc,'diagnostic',{}))
                diagnostic['write_attempts'] = attempt+1
                exc.diagnostic = diagnostic
                status = diagnostic.get('http_status')
                category = diagnostic.get('dashboard_code')
                retry = (category in transient if category else status in (429,500,502,503,504)) or isinstance(exc,core.TransportError)
                if str(exc) == 'scanner_storage_cooldown': raise
                if not retry or attempt==2: raise
                delay = max(diagnostic.get('retry_after_seconds',0),2**attempt) + random.uniform(0,2**attempt)
                try:
                    remaining = self.http.budget.remaining()
                except core.SafeError:
                    diagnostic['retry_deferred'] = 'delivery_budget_insufficient'
                    raise exc
                if remaining <= delay+1:
                    diagnostic['retry_deferred'] = 'delivery_budget_insufficient'
                    raise
                time.sleep(delay)

    def ingest(self, kind, rows):
        if kind not in ('posts','groups'): raise core.SafeError('unsupported_scanner_ingest_kind')
        values = ([{k:row[k] for k in ('platform','account_id','post_id','fingerprint')}
                  for row in rows if row.get('fingerprint')] if kind=='posts' else rows)
        count = 0
        for i in range(0, len(values), 10):
            batch=values[i:i+10]
            try:
                ack=self.write_batch('/api/ingest/scanner/fingerprints' if kind=='posts' else '/api/ingest/duplicate-groups',
                                     {'rows':batch} if kind=='posts' else {'groups':batch})
                if ack.get('accepted')!=len(batch): raise core.SafeError('dashboard_ingest_unacknowledged')
            except core.SafeError as exc:
                raise core.IngestError(str(exc), count, kind, getattr(exc,'diagnostic',{})) from None
            count += len(batch)
        return count

class CatalogueProvider:
    catalogue = True
    def __init__(self, cfg, client):
        self.cfg,self.client=cfg,client
        self.media={}
        self.post_ids=None

    def page(self, after, limit):
        if limit!=20: raise core.SafeError('catalogue_page_size_invalid')
        params={'platform':self.cfg['platform'],'account_id':self.cfg['account_id']}
        if self.post_ids is not None:
            if after:raise core.SafeError('invalid_targeted_posts')
            params['post_ids']=','.join(self.post_ids)
        elif after: params['after']=after
        data=read_bridge(self.client,'posts',params)
        if data.get('source')!='dashboard_catalogue' or not isinstance(data.get('rows'),list):
            raise core.SafeError('dashboard_catalogue_invalid')
        if self.post_ids is not None:
            ids=[str(r.get('post_id','')) for r in data['rows']]
            if set(ids)!=set(self.post_ids) or len(ids)!=len(self.post_ids) or data.get('next_after'):
                raise core.SafeError('targeted_posts_missing_or_mismatched')
        return data['rows'],data.get('next_after')

    def resolve(self, post_id, refresh=False):
        if post_id in self.media and not refresh:return self.media[post_id]
        params={'platform':self.cfg['platform'],'account_id':self.cfg['account_id'],'post_id':post_id}
        if refresh:params['refresh']='1'
        data=read_bridge(self.client,'media',params)
        self.media[post_id]=data
        return data

    def normalize(self, raw, cfg):
        if raw.get('platform')!=cfg['platform'] or raw.get('account_id')!=cfg['account_id']:
            raise core.SafeError('catalogue_identity_mismatch')
        pid=str(raw.get('post_id',''))
        if not core.re.fullmatch(r'[A-Za-z0-9_-]{1,150}',pid):raise core.SafeError('invalid_post_metadata')
        kind=raw.get('scan_kind')
        if kind not in ('IMAGE','VIDEO','CAROUSEL'):
            if raw.get('media_type')!='POST':return None
            kind=self.resolve(pid).get('media_type')
        if kind not in ('IMAGE','VIDEO','CAROUSEL'):raise core.SafeError('original_media_type_unavailable')
        link=core.permalink(raw.get('permalink'),cfg['platform'])
        if not link:raise core.SafeError('missing_or_unsupported_permalink')
        metadata={'platform':cfg['platform'],'account_id':cfg['account_id'],'post_id':pid,'media_type':kind,
                  'permalink':link,'account_handle':core.safe_text(raw.get('account_handle'),150),
                  'published_at':raw.get('published_at'),'caption':core.safe_text(raw.get('caption')), 'metrics':{}}
        return core.Post(metadata)

    def media_url(self, post):
        data=self.resolve(post.metadata['post_id'])
        if data.get('available') is not True or not data.get('url'):raise core.SafeError('original_media_unavailable')
        expected=post.metadata['media_type']
        if data.get('media_type')!=expected or data.get('kind')!=('video' if expected=='VIDEO' else 'image'):
            raise core.SafeError('original_media_type_mismatch')
        return data['url']

    def download(self, post, url, http):
        def get(value):return http.request(value,core.CDNS[self.cfg['platform']],suffix=True,redirects=3,
                                          limit=(10 if post.metadata['media_type']=='IMAGE' else 40)*1024*1024)
        try:return get(url)
        except core.GraphHTTPError as exc:
            if exc.diagnostic.get('http_status') not in (403,404):raise
            self.resolve(post.metadata['post_id'],refresh=True)
            return get(self.media_url(post))

    @staticmethod
    def checkpoint(raw):return {'after':raw['post_id'],'skip':0}

def scan_catalogue(request):
    if request.max_posts>20:raise core.SafeError('maximum_20_catalogue_posts')
    selected=request.post_ids
    if selected is not None and (request.cursor or not selected or len(selected)>20 or len(set(selected))!=len(selected) or request.max_posts!=len(selected) or any(not core.re.fullmatch(r'[A-Za-z0-9_-]{1,150}',p) for p in selected)):
        raise core.SafeError('invalid_targeted_posts')
    cfg={'platform':request.platform,'account_id':request.account_id,'handle':request.account_id,'provider_id':request.account_id}
    client=bridge_client()
    provider=CatalogueProvider(cfg,client)
    provider.post_ids=selected
    sink=FingerprintSink(client.http)
    result=core.scan(request,cfg=cfg,provider=provider,dashboard=sink,normalizer=provider.normalize,cursor_scope='catalogue-v1:')
    if not result['ok']:
        # Keep every failed batch replayable, including failures before ingestion.
        result['resume_cursor'] = request.cursor
        result['coverage']['provider_exhausted'] = False
        result['coverage']['pagination_truncated'] = True
    result['retry_after_seconds'] = max([0] + [float(e.get('diagnostic', {}).get('retry_after_seconds', 0)) for e in result['errors']])
    result['next_batch_after_seconds'] = 75
    coverage=result['coverage']
    coverage['catalogue_exhausted']=coverage.pop('provider_exhausted')
    coverage['provider_exhausted']=False
    coverage['source']='dashboard_catalogue'
    if coverage.get('stop_reason')=='provider_exhausted':coverage['stop_reason']='catalogue_exhausted'
    result['counts']['fingerprints_saved']=result['counts'].pop('posts_ingested')
    result['counts']['metric_observations_created']=0
    if selected is not None:
        coverage['targeted_posts']=True
        coverage['requested_posts']=len(selected)
        coverage['catalogue_exhausted']=False
        if result['ok']:coverage['stop_reason']='targeted_batch_finished'
        result['resume_cursor']=None
    return result
