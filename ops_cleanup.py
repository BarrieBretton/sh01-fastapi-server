"""Receipt reconciliation and opt-in scheduled B2 version cleanup.
No social deletion calls. Every B2 attempt is durably recorded before dispatch.
"""
from __future__ import annotations
import asyncio, hashlib, json, os, threading, time, uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from clickouts_routes import authorize, configured_workbook
from clickouts_state import google_store
from clickouts_sheet_state import SheetsState
from clickouts_worker import post_key, validate, status
from ops_infrastructure import B2, control_key

router = APIRouter()

@router.get('/social/clickouts/evidence', dependencies=[Depends(authorize)])
def social_evidence():
    store = google_store(configured_workbook())
    rows = store.read()
    state = SheetsState(store)
    state.load()
    counts = {}
    for r in rows:
        k = r.get('request_id', '')
        counts[k] = counts.get(k, 0) + 1
    result = []
    for r in rows:
        item = {k: r.get(k, '') for k in ('request_id','platform','account_id','post_id','urls','status','error','deleted_at','dashboard_ack')}
        item['receipt_verified'] = False
        try:
            validate(r)
            receipt = state.get(post_key(r))
            item['receipt_verified'] = bool(counts[r['request_id']] == 1 and status(r) == 'done' and receipt and receipt.get('status') == 'done' and receipt.get('post_id') == r['post_id'] and receipt.get('deleted_at'))
        except Exception:
            pass
        result.append(item)
    return {'rows':result,'checked_at':datetime.now(timezone.utc).isoformat(),'read_only':True}

class CleanupState(SheetsState):
    TITLE = '_sh01_b2_cleanup_state'
    MARKER = 'sh01_b2_cleanup_v1'
    RANGE_NAME = 'SH01_B2_CLEANUP_REVISION'
    def batch(self, requests):
        requests = list(requests)
        for r in list(requests):
            if 'addSheet' in r:
                requests.append({'addProtectedRange':{'protectedRange':{'range':{'sheetId':r['addSheet']['properties']['sheetId']},'description':'Internal B2 cleanup state; excluded from references','warningOnly':True}}})
        return super().batch(requests)

def cleanup_state():
    return CleanupState(google_store(configured_workbook()))

def key_for(slot):
    return hashlib.sha256(slot.encode()).hexdigest()[:32]

def defaults(slot):
    return {'slot':slot,'enabled':False,'top_k':3,'time':'13:30','timezone':'Asia/Kolkata','revision':None}

class Settings(BaseModel):
    slot: str = Field(min_length=1,max_length=120)
    enabled: bool = False
    top_k: int = Field(default=3,ge=1,le=25)
    time: str = Field(default='13:30',pattern=r'^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')
    timezone: str = Field(default='Asia/Kolkata',max_length=80)
    revision: str | None = None

@router.get('/infra/desk/b2/automation', dependencies=[Depends(control_key)])
def settings_get(slot: str):
    if not slot or len(slot)>120: raise HTTPException(400,'invalid_slot')
    s=cleanup_state(); s.load(); k=key_for(slot)
    lease=(s.control or {}).get('lease') or {}
    for name,run in s.values.items():
        if name.startswith('run:') and run.get('status')=='running' and (lease.get('owner')!=run.get('owner') or lease.get('until',0)<=time.time()):
            run['status']='interrupted_review_required'
    return {'settings':s.get('config:'+k) or defaults(slot),'attempts': [v for n,v in s.values.items() if n.startswith('version:') and v.get('slot')==slot][-100:], 'runs':sorted([v for n,v in s.values.items() if n.startswith('run:') and v.get('slot')==slot],key=lambda r:r['started_at'],reverse=True)[:20], 'execution':'automatic_when_enabled','scheduler':'Cloudflare scheduled trigger; runs at next available tick after daily time'}

@router.post('/infra/desk/b2/automation', dependencies=[Depends(control_key)])
def settings_save(value: Settings):
    try: ZoneInfo(value.timezone)
    except Exception: raise HTTPException(400,'invalid_timezone') from None
    # Resolve configuration before enabling; does not contact/delete B2 objects.
    from ops_infrastructure import b2_config
    if value.enabled: b2_config(value.slot)
    s=cleanup_state();s.initialize();k='config:'+key_for(value.slot)
    old=s.get(k) or defaults(value.slot)
    if old['revision']!=value.revision:raise HTTPException(409,'schedule_changed_reload')
    data=value.model_dump(); data['revision']=uuid.uuid4().hex
    # CAS commit may intentionally invalidate an active worker snapshot. Its next
    # durable pre-delete intent must fail before any further provider deletion.
    s.commit({k:data})
    return {'settings':data}

async def inventory(b):
    rows=[]; seen=set(); cursors=set();args={'bucketId':b.bucket,'maxFileCount':1000,'prefix':b.prefix}
    for _ in range(10):
        d=await b.call('b2_list_file_versions',args)
        for f in d['files']:
            r=b.row(f)
            if r['file_id'] in seen:raise RuntimeError('duplicate_inventory_version')
            seen.add(r['file_id']);rows.append(r)
        if not d.get('nextFileName'):return rows
        cursor=(d['nextFileName'],d.get('nextFileId'))
        if cursor in cursors:raise RuntimeError('inventory_cursor_repeated')
        cursors.add(cursor);args.update(startFileName=cursor[0],startFileId=cursor[1])
    raise RuntimeError('inventory_incomplete')

async def desk_call(client,path,payload=None):
    from social_inventory import dashboard_url
    from ops_infrastructure import origin
    base=origin(dashboard_url())
    r=await client.request('POST' if payload is not None else 'GET',base+'/api/ingest/b2-cleanup/'+path,headers={'X-INGEST-KEY':os.environ['CONTROL_PLANE_API_KEY']},**({'json':payload} if payload is not None else {}))
    if r.status_code!=200:raise RuntimeError('workbook_scan_http_'+str(r.status_code))
    return r.json()

def scan_batches(sheets):
    batches=[];batch=[];cells=0
    for sheet in sorted(sheets,key=lambda s:(not s.get('designated',False),s['id'])):
        if sheet['mode']=='excluded':continue
        if sheet['type']!='GRID' or not sheet['rows'] or not 0<sheet['columns']<=20000:raise RuntimeError('sheet_unscannable')
        start=0
        while start<sheet['rows']:
            end=min(sheet['rows'],start+max(1,50000//sheet['columns']))
            cost=(end-start)*sheet['columns']+(sheet['columns'] if not start<sheet['header_row']<=end else 0)
            if len(batch)>=4 or cells+cost>160000 or any(c['sheet_id']==sheet['id'] for c in batch):
                batches.append(batch);batch=[];cells=0
            batch.append({'sheet_id':sheet['id'],'start_row':start,'end_row':end,'rows':sheet['rows']});cells+=cost;start=end
    if batch:batches.append(batch)
    if len(batches)>100:raise RuntimeError('scan_request_limit')
    return batches

async def scan(client,b,objects,state):
    started=time.monotonic();meta=await desk_call(client,'workbook');refs=[];issues=list(meta.get('issues',[]))
    if meta.get('workbook_id')!=configured_workbook():raise RuntimeError('workbook_configuration_mismatch')
    for chunks in scan_batches(meta['sheets']):
        state.guard()
        if time.monotonic()-started>480:raise RuntimeError('scan_expired')
        d=await desk_call(client,'scan',{'fingerprint':meta['fingerprint'],'bucket':b.name,'objects':[{'file_id':r['file_id'],'name':r['name']} for r in objects],'chunks':[{'sheet_id':c['sheet_id'],'start_row':c['start_row']} for c in chunks]})
        results=d.get('results')
        if d.get('fingerprint')!=meta['fingerprint'] or not isinstance(results,list) or len(results)!=len(chunks) or len({r.get('sheet_id') for r in results})!=len(chunks):raise RuntimeError('scan_batch_incomplete')
        for chunk in chunks:
            r=next((r for r in results if r.get('sheet_id')==chunk['sheet_id']),None)
            nxt=chunk['end_row'] if chunk['end_row']<chunk['rows'] else None
            if not r or r.get('fingerprint')!=meta['fingerprint'] or r.get('start_row')!=chunk['start_row'] or r.get('end_row')!=chunk['end_row'] or r.get('next_row')!=nxt or not isinstance(r.get('references'),list) or not isinstance(r.get('issues'),list):raise RuntimeError('scan_batch_incomplete')
        for r in results:refs.extend(r['references']);issues.extend(r['issues'])
        await asyncio.sleep(3) # Preserve the existing quota headroom for n8n.
    end=await desk_call(client,'workbook')
    if end.get('workbook_id')!=configured_workbook() or end['fingerprint']!=meta['fingerprint']:raise RuntimeError('workbook_changed')
    return refs,issues

def eligible(row,refs,issues):
    if row['action']!='upload':return False
    if any(i.get('scope')!='object_key' or i.get('object_name')==row['name'] for i in issues):return False
    matches=[r for r in refs if r['file_id']==row['file_id']]
    return all(r.get('sheet') in ('streamables','streamables-yt') and r.get('designated') is True and r.get('status')=='uploaded' and not r.get('status_problem') and not r.get('uncertain') for r in matches)

def choose(rows,refs,issues,k):
    return sorted([r for r in rows if eligible(r,refs,issues)],key=lambda r:(-r['size'],r['name'],r['file_id']))[:k]

def due(config,now):
    if not config.get('enabled'):return None
    local=now.astimezone(ZoneInfo(config['timezone']))
    if local.strftime('%H:%M')<config['time']:return None
    return local.strftime('%Y-%m-%d')+':'+config['time']

async def execute(s,config,occurrence):
    k=key_for(config['slot']);runkey='run:'+k+':'+occurrence
    if s.get(runkey):return
    run={'owner':getattr(s,'owner',None),'slot':config['slot'],'occurrence':occurrence,'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','results':[],'bytes_reclaimed':0,'settings':config}
    s.put(runkey,run) # Persistent occurrence claim: a restart never replays it.
    try:
        async with httpx.AsyncClient(timeout=35,follow_redirects=False) as client:
            b=await B2.connect(client,config['slot'])
            if not all(c in b.capabilities for c in ('readFiles','listFiles','deleteFiles')):raise RuntimeError('b2_capabilities_missing')
            rows=await inventory(b)
            refs,issues=await scan(client,b,rows,s)
            selected=choose(rows,refs,issues,config['top_k'])
            run['selected_count']=len(selected);run['inventory_count']=len(rows)
            for i,row in enumerate(selected):
                # Fresh complete reference read for every subsequent version.
                if i:refs,issues=await scan(client,b,rows,s)
                if not eligible(row,refs,issues):
                    run['results'].append({'file_id':row['file_id'],'name':row['name'],'status':'skipped_protected'});continue
                if not s.load() or s.get('config:'+k)!=config:raise RuntimeError('schedule_changed_or_paused')
                s.guard()
                attempt='version:'+hashlib.sha256((b.bucket+':'+row['file_id']).encode()).hexdigest()
                if s.get(attempt):
                    run['results'].append({'file_id':row['file_id'],'name':row['name'],'status':'skipped_previous_attempt'});continue
                await b.verify(row)
                intent={'slot':config['slot'],'file_id':row['file_id'],'name':row['name'],'size':row['size'],'status':'outcome_unknown','occurrence':occurrence,'checked_at':datetime.now(timezone.utc).isoformat(),'references':[{k:r.get(k) for k in ('sheet','column','row','cell_url','status')} for r in refs if r['file_id']==row['file_id']],'eligibility':'complete_scan_exact_uploaded_or_unreferenced'}
                # Atomic intent persisted BEFORE DELETE. Any ambiguous write aborts.
                s.put(attempt,intent)
                run['results'].append(intent);s.put(runkey,run)
                result=await b.call('b2_delete_file_version',{'fileName':row['name'],'fileId':row['file_id']})
                if result.get('fileId')!=row['file_id'] or result.get('fileName')!=row['name']:raise RuntimeError('delete_outcome_unknown')
                intent['status']='deleted';s.put(attempt,intent)
                run['bytes_reclaimed']+=row['size'];s.put(runkey,run)
            run['status']='completed'
    except Exception as e:
        run['status']='stopped_review_required'
        # Only static error categories, never URLs/credentials/provider bodies.
        run['error']=str(e) if isinstance(e,RuntimeError) and str(e).replace('_','').isalnum() else type(e).__name__
    finally:
        run['finished_at']=datetime.now(timezone.utc).isoformat()
        s.put(runkey,run)

_lock=threading.Lock()
def tick_worker():
    if not _lock.acquire(False):return
    s=None
    try:
        s=cleanup_state()
        if not s.load():return # No state creation on an unconfigured tick.
        configs=[v for k,v in s.values.items() if k.startswith('config:')]
        for c in configs:
            occurrence=due(c,datetime.now(timezone.utc))
            if not occurrence or s.get('run:'+key_for(c['slot'])+':'+occurrence):continue
            if not s.acquire():return
            try:asyncio.run(execute(s,c,occurrence))
            finally:s.release()
            break # At most one slot per tick.
    except Exception:
        import logging
        logging.getLogger(__name__).warning('b2_cleanup_tick_stopped_check_run_history')
    finally:
        _lock.release()

@router.post('/infra/desk/b2/automation/tick', dependencies=[Depends(control_key)])
def tick():
    threading.Thread(target=tick_worker,daemon=True).start()
    return {'status':'tick_requested','deletions':'only_enabled_due_schedules'}
