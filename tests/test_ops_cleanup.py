"""Isolated production core tests: no Google, Meta, B2, or credentials."""
import ast, asyncio, copy, hashlib, unittest, types
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
SRC=Path(__file__).resolve().parents[1]/'ops_cleanup.py'
def core():
    names={'eligible','choose','due','key_for','execute','inventory','scan'}
    tree=ast.parse(SRC.read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    g=dict(hashlib=hashlib,asyncio=asyncio,datetime=datetime,timezone=timezone,ZoneInfo=ZoneInfo)
    exec(compile(tree,str(SRC),'exec'),g);return g
class State:
    def __init__(self,c):self.data={'config:'+hashlib.sha256(c['slot'].encode()).hexdigest()[:32]:c};self.writes=[]
    def get(self,k):return copy.deepcopy(self.data.get(k))
    def put(self,k,v):self.writes.append(k);self.data[k]=copy.deepcopy(v)
    def load(self):return True
    def guard(self):pass
class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.g=core();self.row={'file_id':'a','name':'v/a.mp4','action':'upload','size':50}
        self.ref={'file_id':'a','sheet':'streamables','designated':True,'status':'uploaded'}
    def test_rules(self):
        e=self.g['eligible'];r=self.row;ref=self.ref
        self.assertTrue(e(r,[],[]));self.assertTrue(e(r,[ref],[]))
        for status in ['','Uploaded','uploaded ','staged',None]:self.assertFalse(e(r,[dict(ref,status=status)],[]))
        self.assertFalse(e(r,[ref,dict(ref,sheet='Archive',designated=False)],[]))
        self.assertFalse(e(r,[ref,dict(ref,status='processing')],[]))
        self.assertFalse(e(r,[dict(ref,uncertain=True)],[]))
        self.assertFalse(e(dict(r,action='hide'),[ref],[]))
        self.assertFalse(e(r,[],[{'reason':'unresolved'}]))
        self.assertTrue(e(r,[],[{'scope':'object_key','object_name':'different'}]))
    def test_topk(self):
        rows=[self.row,dict(self.row,file_id='b',size=100),dict(self.row,file_id='c',size=500,action='hide')]
        self.assertEqual([r['file_id'] for r in self.g['choose'](rows,[],[],1)],['b'])
    def test_due(self):
        c={'enabled':True,'time':'13:30','timezone':'Asia/Kolkata'}
        self.assertIsNone(self.g['due'](c,datetime(2026,11,18,7,59,tzinfo=timezone.utc)))
        self.assertEqual(self.g['due'](c,datetime(2026,11,18,8,0,tzinfo=timezone.utc)),'2026-11-18:13:30')
        self.assertIsNone(self.g['due'](dict(c,enabled=False),datetime.now(timezone.utc)))
    def runner(self,fail=False,pause=False,change_ref=False):
        g=self.g;c={'slot':'test','revision':'v1','top_k':2};s=State(c);events=[]
        rows=[self.row,dict(self.row,file_id='b',name='v/b.mp4',size=40)]
        class Client:
            def __init__(self,**kw):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
        class B:
            capabilities=['readFiles','listFiles','deleteFiles'];bucket='bucket'
            @classmethod
            async def connect(cls,*a):return cls()
            async def verify(self,row):events.append(('verify',row['file_id']))
            async def call(self,method,p):
                selftest.assertTrue(any(k.startswith('version:') for k in s.data))
                events.append(('delete',p['fileId']))
                if fail:raise TimeoutError()
                return {'fileId':p['fileId'],'fileName':p['fileName']}
        selftest=self;calls=0
        async def scan(*a):
            nonlocal calls
            calls+=1
            if pause:s.data['config:'+g['key_for']('test')]=dict(c,revision='paused')
            return ([dict(self.ref,file_id='b',status='processing')] if change_ref and calls>1 else []),[]
        async def inventory(*a):return rows
        g.update(httpx=types.SimpleNamespace(AsyncClient=Client),B2=B,scan=scan,inventory=inventory)
        asyncio.run(g['execute'](s,c,'day'));asyncio.run(g['execute'](s,c,'day'))
        return s,events,c
    def test_durable_claim_and_no_occurrence_replay(self):
        s,events,c=self.runner();self.assertEqual([e for e in events if e[0]=='delete'],[('delete','a'),('delete','b')])
    def test_timeout_stops_batch_and_no_retry(self):
        s,events,c=self.runner(fail=True);self.assertEqual([e for e in events if e[0]=='delete'],[('delete','a')]);self.assertEqual(next(v for k,v in s.data.items() if k.startswith('version:'))['status'],'outcome_unknown')
    def test_pause_stops_before_delete(self):
        s,events,c=self.runner(pause=True);self.assertFalse(any(e[0]=='delete' for e in events))
    def test_recheck_protects_next_file(self):
        s,events,c=self.runner(change_ref=True);self.assertEqual([e for e in events if e[0]=='delete'],[('delete','a')])
if __name__=='__main__':unittest.main()

class EndpointTests(unittest.TestCase):
    def test_auth_and_paused_defaults(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import ops_cleanup
        from unittest.mock import patch
        app=FastAPI();app.include_router(ops_cleanup.router)
        with patch.dict('os.environ',{'CONTROL_PLANE_API_KEY':'control','MEDIA_CLIP_API_KEY':'media'}):
            c=TestClient(app)
            self.assertEqual(c.get('/infra/desk/b2/automation?slot=x').status_code,401)
            self.assertEqual(c.post('/infra/desk/b2/automation/tick').status_code,401)
            self.assertEqual(c.get('/social/clickouts/evidence',headers={'X-API-KEY':'control'}).status_code,401)
        self.assertFalse(ops_cleanup.defaults('x')['enabled'])
    def test_receipt_requires_exact_identity_and_done_status(self):
        import ops_cleanup
        from unittest.mock import patch
        row={'request_id':'r','platform':'instagram','account_id':'a','post_id':'p','urls':'u','status':'done'}
        store=types.SimpleNamespace(read=lambda:[row])
        state=types.SimpleNamespace(load=lambda:True,get=lambda k:{'status':'done','post_id':'wrong','deleted_at':'now'})
        with patch.object(ops_cleanup,'configured_workbook',return_value='book'),patch.object(ops_cleanup,'google_store',return_value=store),patch.object(ops_cleanup,'SheetsState',return_value=state),patch.object(ops_cleanup,'validate'):
            self.assertFalse(ops_cleanup.social_evidence()['rows'][0]['receipt_verified'])
            state.get=lambda k:{'status':'done','post_id':'p','deleted_at':'now'}
            self.assertTrue(ops_cleanup.social_evidence()['rows'][0]['receipt_verified'])
            store.read=lambda:[row,row]
            self.assertFalse(ops_cleanup.social_evidence()['rows'][0]['receipt_verified'])

class BatchScanTests(unittest.TestCase):
    def test_bounds_and_coverage(self):
        import ops_cleanup
        sheets=[{'id':i,'title':str(i),'mode':'included','type':'GRID','rows':1000,'columns':26,'header_row':1} for i in range(28)]
        batches=ops_cleanup.scan_batches(sheets)
        self.assertEqual(len(batches),7)
        self.assertEqual(len([c for b in batches for c in b]),28)
        sheets[0]['rows']=6000
        batches=ops_cleanup.scan_batches(sheets)
        chunks=[c for b in batches for c in b if c['sheet_id']==0]
        self.assertEqual(sum(c['end_row']-c['start_row'] for c in chunks),6000)
        self.assertTrue(all(len({c['sheet_id'] for c in b})==len(b) for b in batches))
    def test_incomplete_batch_fails_closed_and_keeps_three_second_pacing(self):
        import ops_cleanup
        from unittest.mock import AsyncMock,patch
        sheets=[{'id':1,'title':'streamables','mode':'included','type':'GRID','rows':2,'columns':2,'header_row':1}]
        meta={'workbook_id':'book','sheets':sheets,'fingerprint':'f'}
        result={'fingerprint':'f','results':[{'sheet_id':1,'fingerprint':'f','start_row':0,'end_row':2,'next_row':None,'references':[],'issues':[]}]}
        with patch.object(ops_cleanup,'configured_workbook',return_value='book'),patch.object(ops_cleanup,'desk_call',AsyncMock(side_effect=[meta,result,meta])),patch.object(ops_cleanup.asyncio,'sleep',AsyncMock()) as sleep:
            asyncio.run(ops_cleanup.scan(None,types.SimpleNamespace(name='bucket'),[],types.SimpleNamespace(guard=lambda:None)))
            sleep.assert_awaited_once_with(3)
        result['results']=[]
        with patch.object(ops_cleanup,'configured_workbook',return_value='book'),patch.object(ops_cleanup,'desk_call',AsyncMock(side_effect=[meta,result])):
            with self.assertRaisesRegex(RuntimeError,'scan_batch_incomplete'):
                asyncio.run(ops_cleanup.scan(None,types.SimpleNamespace(name='bucket'),[],types.SimpleNamespace(guard=lambda:None)))
