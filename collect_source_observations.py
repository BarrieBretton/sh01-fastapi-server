"""Collect explicitly mapped Drive inventory, or submit explicit source activity.

Config contains source IDs, folder IDs and environment-variable NAMES, never tokens.
No source routing is inferred. Counts are inventory observations, not remaining stock.
"""
import argparse,json,os
from datetime import datetime,timezone
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,build_opener
from read_only_client import NoRedirect

def drive_inventory(config, request):
    folder=config['folder_id']
    if not isinstance(folder,str) or not all(c.isalnum() or c in '_-' for c in folder):raise ValueError('Invalid folder ID')
    count=0;cursor=None;seen=set()
    for _ in range(10):
        q={'q':f"'{folder}' in parents and trashed = false",'fields':'nextPageToken,files(id)','pageSize':100,'supportsAllDrives':'true','includeItemsFromAllDrives':'true'}
        if cursor:q['pageToken']=cursor
        page=request('https://www.googleapis.com/drive/v3/files?'+urlencode(q),os.environ[config['token_env']])
        if not isinstance(page.get('files'),list):raise ValueError('Invalid inventory response')
        count+=len(page['files']);cursor=page.get('nextPageToken')
        if not cursor:return {'count':count,'complete':True,'status':'available','reason':'probe_succeeded'}
        if cursor in seen:raise ValueError('Inventory cursor did not advance')
        seen.add(cursor)
    return {'count':count,'complete':False,'status':'partial','reason':'page_limit'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--submit',action='store_true');a=p.parse_args()
    config=json.load(open(a.config,encoding='utf8'));opener=build_opener(NoRedirect());observations=[]
    def request(url,token):
        with opener.open(Request(url,headers={'Authorization':'Bearer '+token,'User-Agent':'SH01-Source-Inventory/1'}),timeout=20) as r:return json.load(r)
    for c in config['sources']:
        if c['kind']!='google_drive':raise SystemExit('Only explicit Drive collection is supported; submit other source activity from its producing workflow.')
        try:r=drive_inventory(c,request)
        except Exception:r={'count':None,'complete':False,'status':'unavailable','reason':'provider_rejected'}
        observations.append({'source_id':c['source_id'],'measurement':'inventory','checked_at':datetime.now(timezone.utc).isoformat(),**r})
    if a.submit:
        host=os.environ['OPS_DASHBOARD_URL'].rstrip('/');u=urlsplit(host)
        if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment or u.path:raise SystemExit('HTTPS dashboard origin required')
        req=Request(host+'/api/ingest/source-observations',json.dumps({'observations':observations}).encode(),headers={'Content-Type':'application/json','X-INGEST-KEY':os.environ['OPS_INGEST_TOKEN'],'User-Agent':'SH01-Source-Inventory/1'},method='POST')
        try:
            with opener.open(req,timeout=45) as r:json.load(r)
        except Exception:raise SystemExit('Source observation ingestion failed; no response body printed') from None
    print(json.dumps({'submitted':a.submit,'observations':observations},indent=2))
if __name__=='__main__':main()
