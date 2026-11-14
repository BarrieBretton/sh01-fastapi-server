"""Fingerprint dashboard-owned catalogue rows; never call Meta from Render."""
from urllib.parse import urlencode
import social_inventory as core

def bridge_client():
    return core.Dashboard(core.SafeHTTP(core.Budget(seconds=60, max_bytes=8*1024*1024)))

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
    def ingest(self, kind, rows):
        if kind != 'posts': return super().ingest(kind, rows)
        values = [{k:row[k] for k in ('platform','account_id','post_id','fingerprint')}
                  for row in rows if row.get('fingerprint')]
        count = 0
        for i in range(0, len(values), 10):
            batch=values[i:i+10]
            try:
                ack=self.http.json(self.base+'/api/ingest/scanner/fingerprints',self.hosts,
                                   method='POST',headers=self.headers,payload={'rows':batch})
                if ack.get('accepted')!=len(batch): raise core.SafeError('dashboard_ingest_unacknowledged')
            except core.SafeError as exc:
                raise core.IngestError(str(exc), count, kind) from None
            count += len(batch)
        return count

class CatalogueProvider:
    catalogue = True
    def __init__(self, cfg, client):
        self.cfg,self.client=cfg,client
        self.media={}

    def page(self, after, limit):
        if limit!=20: raise core.SafeError('catalogue_page_size_invalid')
        params={'platform':self.cfg['platform'],'account_id':self.cfg['account_id']}
        if after: params['after']=after
        data=read_bridge(self.client,'posts',params)
        if data.get('source')!='dashboard_catalogue' or not isinstance(data.get('rows'),list):
            raise core.SafeError('dashboard_catalogue_invalid')
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
    cfg={'platform':request.platform,'account_id':request.account_id,'handle':request.account_id,'provider_id':request.account_id}
    client=bridge_client()
    provider=CatalogueProvider(cfg,client)
    sink=FingerprintSink(client.http)
    result=core.scan(request,cfg=cfg,provider=provider,dashboard=sink,normalizer=provider.normalize,cursor_scope='catalogue-v1:')
    coverage=result['coverage']
    coverage['catalogue_exhausted']=coverage.pop('provider_exhausted')
    coverage['provider_exhausted']=False
    coverage['source']='dashboard_catalogue'
    if coverage.get('stop_reason')=='provider_exhausted':coverage['stop_reason']='catalogue_exhausted'
    result['counts']['fingerprints_saved']=result['counts'].pop('posts_ingested')
    result['counts']['metric_observations_created']=0
    return result
