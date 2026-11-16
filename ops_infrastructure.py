"""Read-only infrastructure inventory and explicit, version-scoped B2 administration.
No background polling, failover, media downloads, or provider mutation on reads.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field


def now():
    return datetime.now(timezone.utc).isoformat()


def require_key(name, given):
    expected = os.getenv(name, '').strip()
    if not expected:
        raise HTTPException(503, 'monitor_key_not_configured')
    if not given or not hmac.compare_digest(given, expected):
        raise HTTPException(401, 'unauthorized')


def control_key(x_api_key: str | None = Header(None)):
    require_key('CONTROL_PLANE_API_KEY', x_api_key)


def media_key(x_api_key: str | None = Header(None)):
    require_key('MEDIA_CLIP_API_KEY', x_api_key)


router = APIRouter(prefix='/infra/desk', dependencies=[Depends(control_key)])
runtime_router = APIRouter(prefix='/ops/infra', dependencies=[Depends(media_key)])


def present(*names):
    return all(os.getenv(n, '').strip() for n in names)


def origin(value):
    u = urlsplit(value)
    if u.scheme != 'https' or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise HTTPException(503, 'invalid_monitor_url')
    return value.rstrip('/')


def metric(label, used, limit=None, source='measured', unit='bytes'):
    return {'label': label, 'used': used, 'limit': limit, 'unit': unit, 'source': source}


def budget(component, field="storage_bytes"):
    try:
        value = json.loads(os.getenv('OPS_INFRA_LIMITS_JSON', '{}')).get(component, {}).get(field)
        return int(value) if value is not None and int(value) > 0 else None
    except (ValueError, TypeError, AttributeError):
        return None


def card(id, name, configured=True, probe=True, note=''):
    return {'id': id, 'name': name, 'status': 'unchecked' if configured else 'not_configured',
            'can_probe': bool(configured and probe), 'note': note, 'metrics': [],
            'quota_note': 'Provider usage / plan quota is not available from this adapter.', 'checked_at': None}


def runtime_cards():
    # Configuration presence is deliberately not treated as provider health.
    return [
        card('runtime', 'Responding media instance · RAM & temporary disk'),
        card('b2:@default', 'Backblaze B2 · media instance default', present('BACKBLAZE_KEY_ID','BACKBLAZE_APPLICATION_KEY','BACKBLAZE_BUCKET_NAME'), note='Manage files through the matching control-plane B2 slot.'),
        card('queue', 'Cloudflare D1 · media queue', present('CONTROL_PLANE_BASE_URL', 'MEDIA_JOB_STORE_API_KEY')),
        card('redis', 'Redis · legacy scheduler lock', True, note='Uses app REDIS_URL or its n8n-redis default; PING does not verify cron execution.'),
        card('google', 'Google Sheets · operational workbook', bool(present('SPREADSHEET_ID') and (present('GOOGLE_SERVICE_ACCOUNT') or present('GOOGLE_TOKEN_JSON')))),
        card('telegram', 'Telegram Bot API', present('TELEGRAM_BOT_TOKEN')),
        card('youtube', 'YouTube Data API', present('YOUTUBE_API_KEY'), False, 'Key present only; account quota requires Google Cloud monitoring.'),
        card('meta', 'Meta · Facebook / Instagram', bool(present('META_SYSVAR') or present('FB_PAGE_ACCESS_TOKEN')), False, 'Collection evidence is available in Content & analytics; no publishing/deletion permission inferred.'),
        card('imagekit', 'ImageKit · image storage', present('IMAGEKIT_PRIVATE_KEY')),
        card('kite', 'Kite Connect', present('KITE_API_KEY', 'KITE_ACCESS_TOKEN')),
        card('n8n', 'n8n · workflow API', present('N8N_BASE_URL', 'N8N_API_KEY')),
        card('threads', 'Threads · publishing & token store', present('THREADS_ACCOUNTS_JSON'), False, 'Account registry presence only; token expiry and publishing health not probed.'),
        card('tumblr', 'Tumblr · publishing', bool(present('TUMBLR_ACCOUNTS_JSON') or any(k.startswith('TUMBLR_BLOG_IDENTIFIER_') and v for k,v in os.environ.items())), False, 'Account configuration only; OAuth grants not probed.'),
        card('x', 'X · publishing', bool(present('X_ACCOUNTS_JSON') or present('API_KEY','API_KEY_SECRET','ACCESS_TOKEN','ACCESS_TOKEN_SECRET')), False, 'Optional/parked integration; configuration does not prove active use.'),
        card('social_db', 'Social publishing Postgres', present('DB_POSTGRESDB_HOST','DB_POSTGRESDB_USER','DB_POSTGRESDB_PASSWORD')),
        card('cloudinary', 'Cloudinary · historical integration', False, False, 'Historical references; current B2 paths replace storage. No active use inferred.'),
        card('pulsetic', 'Pulsetic · external monitors', True, False, 'Monitor inventory and results are outside this repository; verify in Pulsetic.'),
    ]


def control_cards():
    from infra.registry import registry
    rows = [card('runtime', 'Responding control-plane instance · RAM & temporary disk')]
    for slot in registry.slots('render'):
        for role in ('n8n', 'sh01'):
            rows.append(card(f'render:{slot}:{role}', f'Render · {slot} / {role}'))
    for slot in registry.slots('postgres'):
        rows.append(card('postgres:'+slot, 'Supabase / Postgres · '+slot))
    for slot in registry.slots('b2'):
        rows.append(card('b2:'+slot, 'Backblaze B2 · '+slot))
    if present('BACKBLAZE_KEY_ID','BACKBLAZE_APPLICATION_KEY','BACKBLAZE_BUCKET_NAME'):
        rows.append(card('b2:@default', 'Backblaze B2 · this instance default'))
    for role in ('sh01', 'n8n'):
        configured = present(role.upper()+'_ROUTER_CONTROL_BASE_URL', role.upper()+'_ROUTER_CONTROL_KEY')
        if role == 'n8n':
            configured = configured or present('ROUTER_CONTROL_BASE_URL','ROUTER_CONTROL_KEY')
        rows.append(card('router:'+role, 'Cloudflare router + KV · '+role, configured))
    rows += [card('sla', 'SLA controller / watchdog evidence'),
             card('cloudflare', 'Cloudflare · account usage, Workers, D1 & KV quotas', True, False,
                  'Router persistence and queue health are probed separately. Account billing/usage and dashboard-managed bindings require Cloudflare account telemetry.')]
    return rows


@runtime_router.get('/inventory')
def runtime_inventory():
    return {'rows': runtime_cards(), 'observed_at': now(), 'scope': 'responding_media_instance',
            'instance': os.getenv('RENDER_SERVICE_ID','local'), 'coverage': 'Configuration is local to the responding instance; standby configurations may differ.'}


@router.get('/inventory')
def inventory():
    from infra.state import state
    return {'rows': control_cards(), 'active': state.all(), 'observed_at': now(),
            'scope': 'control_plane_registry', 'instance': os.getenv('RENDER_SERVICE_ID','local'),
            'coverage': 'All registered slots; external services absent from the live registry are not discoverable. Active means controller-selected, not independently verified traffic.'}


async def request_json(client, method, url, **kwargs):
    response = await client.request(method, url, **kwargs)
    if not 200 <= response.status_code < 300:
        raise HTTPException(502, 'provider_http_'+str(response.status_code))
    return response.json()


async def db_metrics(cfg, component):
    import asyncpg
    conn = await asyncpg.connect(**cfg, timeout=5, command_timeout=5)
    try:
        size = await conn.fetchval('SELECT pg_database_size(current_database())')
        return [metric('Database storage', int(size), budget(component), 'measured / operator budget' if budget(component) else 'measured')]
    finally:
        await conn.close(timeout=2)


def resource_metrics():
    # Raw cgroup usage (including cache) is labeled separately from process RSS.
    rows = []
    for used_path, limit_path in [('/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory.max'),('/sys/fs/cgroup/memory/memory.usage_in_bytes','/sys/fs/cgroup/memory/memory.limit_in_bytes')]:
        try:
            used, raw = int(Path(used_path).read_text()), Path(limit_path).read_text().strip()
            limit = int(raw) if raw != 'max' else None
            if limit and limit >= 1 << 60: limit = None
            rows.append(metric('Container RAM (includes cache)', used, limit))
            break
        except (OSError, ValueError):
            pass
    try:
        rss = next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        rows.append(metric('This Python process RSS', int(rss.split()[1])*1024))
    except (OSError, StopIteration, ValueError):
        pass
    disk = shutil.disk_usage('/tmp')
    rows.append(metric('Temporary filesystem (shared filesystem capacity)', disk.used, disk.total))
    return rows


async def google_probe():
    # Avoid sheets_helper import side effects / interactive OAuth setup.
    def load():
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        raw = os.getenv('GOOGLE_SERVICE_ACCOUNT')
        creds = (service_account.Credentials.from_service_account_info(json.loads(raw), scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
                 if raw else Credentials.from_authorized_user_info(json.loads(os.environ['GOOGLE_TOKEN_JSON'])))
        if not creds.valid:
            class TimedRequest(Request):
                def __call__(self, *args, **kwargs):
                    kwargs['timeout'] = 6
                    return super().__call__(*args, **kwargs)
            creds.refresh(TimedRequest())
        return creds.token
    token = await asyncio.to_thread(load)
    async with httpx.AsyncClient(timeout=6, follow_redirects=False) as client:
        from urllib.parse import quote
        data = await request_json(client, 'GET', 'https://sheets.googleapis.com/v4/spreadsheets/'+quote(os.environ['SPREADSHEET_ID'],safe=''),
                                  headers={'Authorization':'Bearer '+token}, params={'fields':'spreadsheetId'})
        if not data.get('spreadsheetId'): raise ValueError('missing sheet')


async def probe(component, runtime=False):
    rows = runtime_cards() if runtime else control_cards()
    item = next((r for r in rows if r['id'] == component), None)
    if item is None: raise HTTPException(404, 'unknown_component')
    if not item['can_probe']: return item
    started = time.monotonic()
    try:
        async with asyncio.timeout(18):
            async with httpx.AsyncClient(timeout=6, follow_redirects=False) as client:
                if component == 'runtime':
                    item['metrics'] = resource_metrics()
                    item['note'] = 'Only this responding container, not all Render services or monthly compute allowance.'
                elif component.startswith('render:'):
                    from infra.registry import registry
                    _, slot, role = component.split(':',2)
                    cfg = registry.render_service(slot,role)
                    base = origin(os.environ[cfg['base_url_env']])
                    path = cfg.get('health_path', '/healthz/readiness' if role=='n8n' else '/')
                    if not path.startswith('/') or path.startswith('//'): raise ValueError('health path')
                    r = await client.get(base+path)
                    if not 200 <= r.status_code < 300: raise HTTPException(502,'provider_http_'+str(r.status_code))
                    item['note'] = 'Direct origin health responded HTTP '+str(r.status_code)+'. Provider suspension, deploy state and workspace compute allowance are not inferred.'
                elif component.startswith('postgres:'):
                    from infra.postgres import get_postgres_config
                    c = get_postgres_config(component.split(':',1)[1])
                    item['metrics'] = await db_metrics(dict(host=c.host,port=c.port,user=c.user,password=c.password,database=c.database),component)
                elif component == 'social_db':
                    item['metrics'] = await db_metrics(dict(host=os.environ['DB_POSTGRESDB_HOST'], port=int(os.getenv('DB_POSTGRESDB_PORT','6543')),user=os.environ['DB_POSTGRESDB_USER'],password=os.environ['DB_POSTGRESDB_PASSWORD'],database=os.getenv('DB_POSTGRESDB_DATABASE','postgres')),component)
                elif component.startswith('b2:'):
                    b = await B2.connect(client, component.split(':',1)[1])
                    await b.call('b2_list_file_versions', {'bucketId':b.bucket,'maxFileCount':1,'prefix':b.prefix})
                    item['note'] = 'Authorized bucket metadata listing. Open Manage files to measure stored version bytes; no media downloaded.'
                    item['bucket'] = b.name
                elif component.startswith('router:'):
                    role = component.split(':')[1].upper()
                    base = os.getenv(role+'_ROUTER_CONTROL_BASE_URL') or (os.getenv('ROUTER_CONTROL_BASE_URL') if role=='N8N' else '')
                    key = os.getenv(role+'_ROUTER_CONTROL_KEY') or (os.getenv('ROUTER_CONTROL_KEY') if role=='N8N' else '')
                    d = await request_json(client,'GET',origin(base)+'/__control/status',headers={'X-ROUTER-KEY':key})
                    item['note'] = 'Maintenance: '+str(bool(d.get('maintenance')))+'; KV persistence: '+str(d.get('persistence_available','unknown'))
                    if d.get('maintenance') or d.get('persistence_available') is False: item['status']='attention'
                elif component=='queue':
                    d = await request_json(client,'GET',origin(os.environ['CONTROL_PLANE_BASE_URL'])+'/internal/media-jobs/queue/health',headers={'X-MEDIA-JOB-KEY':os.environ['MEDIA_JOB_STORE_API_KEY']})
                    if d.get('ok') is not True: raise ValueError('queue unhealthy')
                    item['note']='Queue health only; D1 account capacity and consumer progress are separate.'
                elif component=='sla':
                    from infra.sla import sla_status
                    d = await asyncio.to_thread(sla_status)
                    monitor = d.get('monitor') or {}
                    last = monitor.get('last_checked_at')
                    age = max(0, (datetime.now(timezone.utc)-datetime.fromisoformat(last.replace('Z','+00:00'))).total_seconds()) if last else None
                    item['status']='attention' if monitor.get('hard_stop') or (age is not None and age>600) else 'unchecked'
                    item['note']='Last recorded controller check: '+(str(last) if last else 'unknown')+'. This read does not execute a tick or prove the external watchdog is running.'
                    item['metrics'] = [metric('Consecutive failed checks',int(monitor.get('consecutive_failures',0)),unit='count')]
                    if age is not None: item['metrics'].append(metric('Age of recorded check',round(age),unit='seconds'))
                elif component=='redis':
                    from redis.asyncio import Redis
                    redis = Redis.from_url(os.getenv('REDIS_URL','redis://n8n-redis:6379/0'),socket_connect_timeout=4,socket_timeout=4)
                    try:
                        await redis.ping()
                        try:
                            info=await redis.info('memory')
                            item['metrics']=[metric('Redis memory',int(info['used_memory']),int(info.get('maxmemory',0)) or None)]
                        except Exception:
                            item['note']+=' Memory INFO unavailable for this ACL.'
                    finally: await redis.aclose()
                elif component=='google': await google_probe()
                elif component=='telegram':
                    d=await request_json(client,'GET','https://api.telegram.org/bot'+os.environ['TELEGRAM_BOT_TOKEN']+'/getMe')
                    if d.get('ok') is not True: raise ValueError('telegram unhealthy')
                elif component=='imagekit':
                    await request_json(client,'GET','https://api.imagekit.io/v1/files',auth=(os.environ['IMAGEKIT_PRIVATE_KEY'],''),params={'limit':1})
                    item['note']='File-list access verified.'
                    try:
                        end=datetime.now(timezone.utc).date();start=end-timedelta(days=30)
                        usage=await request_json(client,'GET','https://api.imagekit.io/v1/accounts/usage',auth=(os.getenv('OPS_IMAGEKIT_USAGE_KEY') or os.environ['IMAGEKIT_PRIVATE_KEY'],''),params={'startDate':str(start),'endDate':str(end)})
                        for field,label,unit,limit_field in [('mediaLibraryStorageBytes','Media library storage','bytes','storage_bytes'),('originalCacheStorageBytes','Original cache storage','bytes','cache_bytes'),('bandwidthBytes','Bandwidth · last 30 complete UTC days','bytes','bandwidth_bytes'),('videoProcessingUnitsCount','Video processing units · last 30 complete UTC days','count','video_units'),('extensionUnitsCount','Extension units · last 30 complete UTC days','count','extension_units')]:
                            if isinstance(usage.get(field),(int,float)) and usage[field]>=0:
                                item['metrics'].append(metric(label,usage[field],budget('imagekit',limit_field),'ImageKit usage API / optional operator budget',unit))
                        item['note']+=' Usage period '+str(start)+' to '+str(end)+' (end exclusive; provider may cache for 6 hours). Agency account data can include billed child accounts.'
                    except Exception:
                        item['note']+=' Usage API unavailable for this key; optional OPS_IMAGEKIT_USAGE_KEY needs account usage read access.'
                elif component=='kite':
                    await request_json(client,'GET','https://api.kite.trade/user/profile',headers={'X-Kite-Version':'3','Authorization':'token '+os.environ['KITE_API_KEY']+':'+os.environ['KITE_ACCESS_TOKEN']})
                elif component=='n8n':
                    await request_json(client,'GET',origin(os.environ['N8N_BASE_URL'])+'/api/v1/workflows',headers={'X-N8N-API-KEY':os.environ['N8N_API_KEY']},params={'limit':1})
                else: raise ValueError('unsupported')
        if item['status']=='unchecked' and component!='sla': item['status']='healthy'
    except HTTPException as exc:
        item['status']='unavailable'; item['note']=str(exc.detail)
    except (TimeoutError, httpx.TimeoutException):
        item['status']='unavailable'; item['note']='probe_timeout'
    except Exception:
        item['status']='unavailable'; item['note']='probe_failed_check_server_configuration_or_provider_access'
    item['checked_at']=now(); item['latency_ms']=round((time.monotonic()-started)*1000)
    return item


# Bounded per-process single-flight cache; caller refresh cannot fan out provider probes.
_cache = {}
_gate = asyncio.Semaphore(3)
async def cached_probe(component, runtime):
    key=(runtime,component)
    old=_cache.get(key)
    if old and old[0] > time.monotonic(): return await asyncio.shield(old[1])
    async def run():
        async with _gate: return await probe(component,runtime)
    task=asyncio.create_task(run())
    _cache[key]=(time.monotonic()+60,task)
    # Only the registry and fixed manifest determine valid cache keys.
    return await asyncio.shield(task)


@router.get('/check')
async def check(component: str = Query(max_length=180)):
    if component not in {r['id'] for r in control_cards()}: raise HTTPException(404,'unknown_component')
    return await cached_probe(component,False)


@runtime_router.get('/check')
async def runtime_check(component: str = Query(max_length=180)):
    if component not in {r['id'] for r in runtime_cards()}: raise HTTPException(404,'unknown_component')
    return await cached_probe(component,True)


def b2_config(slot):
    if slot=='@default':
        names=('BACKBLAZE_KEY_ID','BACKBLAZE_APPLICATION_KEY','BACKBLAZE_BUCKET_NAME')
    else:
        from infra.registry import registry
        if not registry.exists('b2',slot): raise HTTPException(404,'unknown_b2_slot')
        cfg=registry.get('b2',slot)
        names=(cfg['key_id_env'],cfg['application_key_env'],cfg['bucket_name_env'])
    if not present(*names): raise HTTPException(503,'b2_slot_not_configured')
    return tuple(os.environ[n].strip() for n in names)


def seal(payload):
    secret=os.getenv('CONTROL_PLANE_API_KEY','')
    if not secret: raise HTTPException(503,'monitor_key_not_configured')
    raw=base64.urlsafe_b64encode(json.dumps(payload,separators=(',',':'),ensure_ascii=True).encode()).decode().rstrip('=')
    return raw+'.'+hmac.new(secret.encode(),('ops-b2-v1:'+raw).encode(),hashlib.sha256).hexdigest()


def unseal(ticket, kind, slot):
    try:
        raw,sig=ticket.rsplit('.',1)
        expected=hmac.new(os.environ['CONTROL_PLANE_API_KEY'].encode(),('ops-b2-v1:'+raw).encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig,expected): raise ValueError()
        d=json.loads(base64.urlsafe_b64decode(raw+'='*(-len(raw)%4)))
        scope=hashlib.sha256(json.dumps(b2_config(slot)).encode()).hexdigest()
        if d['kind']!=kind or d['slot']!=slot or d['expires']<time.time() or d['scope']!=scope: raise ValueError()
        return d
    except Exception:
        raise HTTPException(409,'b2_review_expired_or_invalid_refresh_listing') from None


class B2:
    def __init__(self,client,base,token,bucket,name,prefix,capabilities,slot,scope):
        self.client,self.base,self.token,self.bucket,self.name,self.prefix,self.capabilities,self.slot,self.scope=client,base,token,bucket,name,prefix,capabilities,slot,scope

    @classmethod
    async def connect(cls,client,slot):
        key,secret,name=b2_config(slot)
        d=await request_json(client,'GET','https://api.backblazeb2.com/b2api/v3/b2_authorize_account',auth=(key,secret))
        storage=d['apiInfo']['storageApi'];base=origin(storage['apiUrl'])
        if not urlsplit(base).hostname.endswith('.backblazeb2.com'): raise HTTPException(502,'invalid_b2_api_host')
        caps=storage.get('capabilities',[])
        bucket=storage.get('bucketId') if storage.get('bucketName')==name else None
        instance=cls(client,base,d['authorizationToken'],bucket,name,storage.get('namePrefix') or '',caps,slot,hashlib.sha256(json.dumps((key,secret,name)).encode()).hexdigest())
        if not bucket:
            result=await instance.call('b2_list_buckets',{'accountId':d['accountId'],'bucketName':name})
            matches=[b for b in result['buckets'] if b['bucketName']==name]
            if len(matches)!=1: raise HTTPException(409,'b2_bucket_not_resolved')
            instance.bucket=matches[0]['bucketId']
        return instance

    async def call(self,method,payload):
        # No automatic retries, especially for destructive requests.
        return await request_json(self.client,'POST',self.base+'/b2api/v3/'+method,headers={'Authorization':self.token},json=payload)

    def ticket(self,kind,**payload):
        return seal(dict(kind=kind,slot=self.slot,scope=self.scope,bucket=self.bucket,expires=int(time.time())+600,**payload))

    def row(self,f):
        return {'file_id':f['fileId'],'name':f['fileName'],'size':int(f.get('contentLength',0)),
                'uploaded_at':int(f.get('uploadTimestamp',0)),'action':f.get('action','unknown')}

    async def verify(self,row):
        f=await self.call('b2_get_file_info',{'fileId':row['file_id']})
        if f.get('bucketId')!=self.bucket or self.row(f)!=row or row['action']!='upload' or not row['name'].startswith(self.prefix):
            raise HTTPException(409,'b2_file_identity_changed')


@router.get('/b2/files')
async def files(slot: str = Query(max_length=120), cursor: str = Query('',max_length=12000)):
    try:
        async with asyncio.timeout(22), httpx.AsyncClient(timeout=6,follow_redirects=False) as client:
            b=await B2.connect(client,slot)
            args={'bucketId':b.bucket,'maxFileCount':500,'prefix':b.prefix}
            if cursor:
                c=unseal(cursor,'cursor',slot)
                if c['bucket']!=b.bucket: raise HTTPException(409,'b2_bucket_changed')
                args.update(startFileName=c['name'],startFileId=c['file_id'])
            d=await b.call('b2_list_file_versions',args)
            rows=[]
            for f in d['files']:
                row=b.row(f)
                row['ticket']=b.ticket('file',row=row) if row['action']=='upload' else None
                rows.append(row)
            rows.sort(key=lambda r:(-r['size'],r['name'],r['file_id']))
            nxt=b.ticket('cursor',name=d['nextFileName'],file_id=d.get('nextFileId')) if d.get('nextFileName') else None
            return {'rows':rows,'next_cursor':nxt,'bucket':b.name,'prefix':b.prefix,'observed_at':now(),
                    'storage_budget':budget('b2:'+slot),'can_delete':all(c in b.capabilities for c in ('readFiles','deleteFiles')),
                    'scope':'All returned versions; sizes exclude unfinished uploads. Partial until pagination completes. Not an atomic bucket snapshot.'}
    except HTTPException: raise
    except Exception: raise HTTPException(502,'b2_listing_unavailable') from None


class Preview(BaseModel):
    slot: str = Field(max_length=120)
    tickets: list[str] = Field(min_length=1,max_length=10)


class Delete(BaseModel):
    slot: str = Field(max_length=120)
    plan: str = Field(max_length=50000)
    confirmation: str = Field(max_length=40)


@router.post('/b2/preview')
async def preview(req: Preview):
    if any(len(t)>12000 for t in req.tickets): raise HTTPException(400,'invalid_ticket_size')
    records=[unseal(t,'file',req.slot) for t in req.tickets]
    rows=[r['row'] for r in records]
    if len({r['file_id'] for r in rows})!=len(rows): raise HTTPException(400,'duplicate_file_selection')
    try:
        async with asyncio.timeout(25), httpx.AsyncClient(timeout=6,follow_redirects=False) as client:
            b=await B2.connect(client,req.slot)
            if not all(c in b.capabilities for c in ('readFiles','deleteFiles')): raise HTTPException(403,'b2_read_and_delete_capabilities_required')
            if any(r['bucket']!=b.bucket for r in records): raise HTTPException(409,'b2_bucket_changed')
            for row in rows: await b.verify(row)
            return {'plan':b.ticket('plan',rows=rows),'rows':rows,'bucket':b.name,'bytes':sum(r['size'] for r in rows),
                    'confirmation':'DELETE '+str(len(rows)),'expires_in':600}
    except HTTPException: raise
    except Exception: raise HTTPException(502,'b2_preview_unavailable') from None


@router.post('/b2/delete')
async def delete(req: Delete):
    plan=unseal(req.plan,'plan',req.slot)
    rows=plan['rows']
    if req.confirmation!='DELETE '+str(len(rows)): raise HTTPException(400,'confirmation_mismatch')
    results=[]
    try:
        async with httpx.AsyncClient(timeout=6,follow_redirects=False) as client:
            b=await B2.connect(client,req.slot)
            if plan['bucket']!=b.bucket: raise HTTPException(409,'b2_bucket_changed')
            if not all(c in b.capabilities for c in ('readFiles','deleteFiles')): raise HTTPException(403,'b2_delete_not_allowed')
            deadline=time.monotonic()+35
            for row in rows:
                result={'file_id':row['file_id'],'name':row['name'],'status':'not_attempted'}
                results.append(result)
                if time.monotonic()>deadline: continue
                try:
                    await b.verify(row)
                except Exception:
                    result['status']='blocked';continue
                try:
                    d=await b.call('b2_delete_file_version',{'fileName':row['name'],'fileId':row['file_id']})
                    result['status']='deleted' if d.get('fileId')==row['file_id'] and d.get('fileName')==row['name'] else 'unknown'
                except Exception:
                    result['status']='unknown' # A transport failure does not prove a delete failed.
                    deadline=0 # Stop subsequent deletes; caller must re-list, not auto-retry.
        return {'results':results,'checked_at':now(),'instruction':'Refresh the listing before a new selection. Unknown outcomes must be reconciled by listing.'}
    except HTTPException: raise
    except Exception: raise HTTPException(502,'b2_delete_setup_failed') from None
