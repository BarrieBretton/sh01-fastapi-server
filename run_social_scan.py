"""Run bounded authenticated scans with a private, atomic resume checkpoint.

No publishing/deletion. Provider media is read; metadata/fingerprints are ingested.
"""
import argparse,json,os,tempfile,time,random
from urllib.error import HTTPError
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,build_opener
from read_only_client import NoRedirect

def throttled_delay(result):
    items = result.get('errors') or [result.get('detail', {})]
    if not items or any(not isinstance(x, dict) for x in items): return None
    delay = 75
    for item in items:
        d = item.get('diagnostic', item)
        if d.get('dashboard_code') != 'store_rate_limited' and item.get('code') != 'scanner_storage_cooldown': return None
        delay = max(delay, float(d.get('retry_after_seconds', 0)))
    return delay

def request_batch(opener, req):
    for attempt in range(3):
        try:
            with opener.open(req, timeout=360) as response: result=json.load(response)
        except HTTPError as exc:
            try: result=json.loads(exc.read(65536))
            except Exception: raise SystemExit('Unreadable scanner error; checkpoint retained.') from None
        except Exception: raise SystemExit('Scan response unavailable; checkpoint retained. No automatic replay of ambiguous timeout.') from None
        if result.get('ok') is True: return result
        delay=throttled_delay(result)
        if delay is None or attempt==2 or delay>300:
            print(json.dumps({'errors':result.get('errors'), 'detail':result.get('detail')}))
            raise SystemExit('Scan failed; checkpoint retained.')
        wait=delay+random.uniform(0,5)
        print('Storage cooldown; retrying the same batch after',round(wait),'seconds.',flush=True)
        time.sleep(wait)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('platform',choices=['facebook','instagram','tumblr']);p.add_argument('account_id');p.add_argument('--batches',type=int,default=1);p.add_argument('--max-posts',type=int,default=6);p.add_argument('--checkpoint',default='PRIVATE-SCAN-CHECKPOINT.json');a=p.parse_args()
    if not 1<=a.max_posts<=20:raise SystemExit('Use 1–20 posts per batch')
    if not 1<=a.batches<=10:raise SystemExit('Use 1–10 bounded batches per invocation')
    host=os.getenv('SOCIAL_API_URL','https://sh01.vivojaymail.workers.dev').rstrip('/');u=urlsplit(host)
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment or u.path:raise SystemExit('HTTPS social API origin required')
    token=os.environ.get('SOCIAL_INVENTORY_API_KEY','').strip() or os.environ.get('MEDIA_CLIP_API_KEY','').strip();path=Path(a.checkpoint)
    if not token:raise SystemExit('Set MEDIA_CLIP_API_KEY locally (or an explicit SOCIAL_INVENTORY_API_KEY override)')
    state=json.loads(path.read_text()) if path.exists() else {'platform':a.platform,'account_id':a.account_id,'cursor':None}
    if (state['platform'],state['account_id'])!=(a.platform,a.account_id):raise SystemExit('Checkpoint belongs to another account')
    if state.get('complete'):raise SystemExit('Checkpoint is complete. Choose a new checkpoint for a new scan.')
    opener=build_opener(NoRedirect())
    for n in range(a.batches):
        req=Request(host+'/social/inventory/scan',json.dumps({'platform':a.platform,'account_id':a.account_id,'cursor':state['cursor'],'max_posts':a.max_posts}).encode(),method='POST',headers={'X-API-KEY':token,'Content-Type':'application/json','User-Agent':'SH01-Scan-Runner/1'})
        result=request_batch(opener,req)
        if not isinstance(result.get('coverage'),dict) or 'resume_cursor' not in result:raise SystemExit('Invalid scan response; checkpoint retained')
        state.update(cursor=result['resume_cursor'],complete=(result['coverage'].get('catalogue_exhausted') is True if result['coverage'].get('source')=='dashboard_catalogue' else result['coverage'].get('provider_exhausted') is True))
        path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.scan-')
        try:
            with os.fdopen(fd,'w') as f:json.dump(state,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        print(json.dumps({'batch':n+1,'counts':result.get('counts',{}),'coverage':result['coverage']}))
        if state['complete'] or not result.get('ok'):break
        if n+1<a.batches:time.sleep(75)
if __name__=='__main__':main()
