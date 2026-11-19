import copy
import json
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import workflow_guards as g
from clickouts_state import GoogleClickouts, ClickoutError
from fastapi import HTTPException

class Request:
    def __init__(self, f): self.f=f
    def execute(self, num_retries=0): return self.f()

class BadRequest(Exception):
    resp=type('Resp', (dict,), {'status':400})()

class API:
    """Atomic in-memory model of the exact Sheets request protocol."""
    def __init__(self):
        self.sheets={}; self.ranges={}; self.before=None; self.lose=False
        self.queue={}; self.writes=0
    def spreadsheets(self): return self
    def get(self, **kwargs):
        def run():
            if kwargs.get('ranges') and not self.sheets: raise BadRequest()
            return {'sheets':[{'properties':copy.deepcopy(s['properties']),
                    'data':[{'startRow':0,'rowData':copy.deepcopy(s['rows'])}]} for s in self.sheets.values()],
                    'namedRanges':copy.deepcopy(list(self.ranges.values()))}
        return Request(run)
    def batchUpdate(self, **kwargs):
        def run():
            if self.before:
                fn,self.before=self.before,None;fn()
            sheets,ranges,queue=copy.deepcopy((self.sheets,self.ranges,self.queue))
            for req in kwargs['body']['requests']:
                if 'addSheet' in req:
                    props=req['addSheet']['properties']; sid=props['sheetId']
                    if any(x['properties']['title']==props['title'] for x in sheets.values()): raise BadRequest()
                    sheets[sid]={'properties':props,'rows':[]}
                elif 'addNamedRange' in req:
                    nr=req['addNamedRange']['namedRange']
                    if nr['namedRangeId'] in ranges or any(x['name']==nr['name'] for x in ranges.values()): raise BadRequest()
                    ranges[nr['namedRangeId']]=nr
                elif 'updateNamedRange' in req:
                    nr=req['updateNamedRange']['namedRange']
                    if nr['namedRangeId'] not in ranges: raise BadRequest()
                elif 'deleteNamedRange' in req:
                    nid=req['deleteNamedRange']['namedRangeId']
                    if nid not in ranges: raise BadRequest()
                    del ranges[nid]
                elif 'updateCells' in req:
                    data=req['updateCells'];start=data['start'];sid=start['sheetId']
                    if sid==999:
                        queue[start['columnIndex']]=data['rows'][0]['values'][0]['userEnteredValue']['stringValue']
                        continue
                    rows=sheets[sid]['rows'];i=start['rowIndex'];col=start['columnIndex']
                    while len(rows)<=i: rows.append({'values':[]})
                    old=rows[i]['values'];new=data['rows'][0]['values']
                    while len(old)<col+len(new):old.append({})
                    old[col:col+len(new)]=new
                else: raise AssertionError(req)
            self.sheets,self.ranges,self.queue=sheets,ranges,queue;self.writes+=1
            if self.lose:
                self.lose=False;raise OSError('lost reply')
            return {}
        return Request(run)

class Tests(unittest.TestCase):
    def setUp(self):
        self.api=API();self.now=1000
        self.headers=['queue_id']+list(g.OWNED)
        self.row={k:'' for k in self.headers}
        self.row.update(queue_id='q1',status='staged',row_number=2)
        self.context=patch.object(g,'context',side_effect=self.fresh);self.context.start()
        self.snapshot=patch.object(g,'snapshot',side_effect=self.snap);self.snapshot.start()
        self.fresh().initialize()
    def tearDown(self): self.context.stop();self.snapshot.stop()
    def fresh(self): return g.GuardState(GoogleClickouts(self.api,'book'),clock=lambda:self.now)
    def snap(self,state,schedule=False):
        if schedule:
            dt=datetime.now(timezone.utc).astimezone(g.IST)
            row={'row_number':2,'workflow':'A','timings':dt.strftime('%I:%M%p'), 'run as job today':'yes'}
            return {'sheetId':g.SCHEDULE_ID},list(row),[row]
        return {'sheetId':999},self.headers,[copy.deepcopy(self.row)]
    def claim(self):return g.queue_claim(g.QueueClaim(queue_id='q1',action='submit'))
    def commit_request(self,claim):
        return g.QueueCommit(queue_id='q1',guard=claim['guard'],changes={'status':'rendering','render_job_id':'av-q1'})
    def test_initialization_verified_and_health_read_only(self):
        before=self.api.writes
        self.assertTrue(g.health()['state_verified']);self.assertEqual(self.api.writes,before)
    def test_duplicate_queue_claim_denied(self):
        self.assertTrue(self.claim()['allowed']);self.assertFalse(self.claim()['allowed'])
    def test_expired_claim_replaced_and_old_commit_rejected(self):
        a=self.claim();self.now+=181;b=self.claim()
        self.assertNotEqual(a['guard'],b['guard'])
        with self.assertRaises(HTTPException):g.queue_commit(self.commit_request(a))
        self.assertEqual(self.api.queue,{})
    def test_changed_sheet_row_rejected(self):
        a=self.claim();self.row['status']='rendered'
        with self.assertRaises(HTTPException):g.queue_commit(self.commit_request(a))
        self.assertEqual(self.api.queue,{})
    def test_commit_and_receipt_share_atomic_batch(self):
        a=self.claim();self.assertTrue(g.queue_commit(self.commit_request(a))['ok'])
        self.assertEqual(self.api.queue[self.headers.index('status')],'rendering')
        self.assertEqual(self.fresh().get('queue:'+g.digest('q1'))['until'],0)
    def test_stale_state_cannot_write_queue(self):
        a=self.claim();stale=self.fresh();stale.load()
        other=self.fresh();other.load();other.commit({'unrelated':{'ok':True}})
        with self.assertRaises(HTTPException):g.commit_queue(stale,self.commit_request(a))
        self.assertEqual(self.api.queue,{})
    def test_lost_reply_reconciles_without_second_batch(self):
        a=self.claim();self.api.lose=True;n=self.api.writes
        self.assertTrue(g.queue_commit(self.commit_request(a))['ok'])
        self.assertEqual(self.api.writes,n+1)
    def test_commit_twice_rejected(self):
        a=self.claim();g.queue_commit(self.commit_request(a))
        with self.assertRaises(HTTPException):g.queue_commit(self.commit_request(a))
    def test_dispatch_is_permanent_and_payload_authoritative(self):
        r=g.Dispatch(row_number=2,workflow_id='A',workflow_name='A',scheduled_at=datetime.now(timezone.utc).isoformat())
        a=g.schedule_claim(r);self.assertTrue(a['allowed']);self.assertTrue(a['payload']['job'])
        self.assertFalse(g.schedule_claim(r)['allowed'])
    def test_two_stale_dispatch_reservations_one_wins(self):
        a=self.fresh();a.load();b=self.fresh();b.load()
        a.commit({'dispatch:x':{'workflow_id':'A'}})
        with self.assertRaises(HTTPException):g.safe_commit(b,{'dispatch:x':{'workflow_id':'B'}})
        self.assertEqual(self.fresh().get('dispatch:x')['workflow_id'],'A')
    def test_old_occurrence_blocked(self):
        r={'row_number':2,'timings':'11:15pm'}
        dt=datetime(2026,11,18,17,45,tzinfo=timezone.utc)
        self.assertTrue(g.occurrence(r,dt.isoformat(),now=dt))
        with self.assertRaises(HTTPException):g.occurrence(r,dt.isoformat(),now=dt+timedelta(minutes=3))
    def test_naive_timestamp_blocked(self):
        with self.assertRaises(HTTPException):g.occurrence({'row_number':2,'timings':'1pm'},'2026-11-18T13:00:00')
    def test_schema_and_identity_changes_blocked(self):
        a=self.claim();self.headers=self.headers+['newcol']
        with self.assertRaises(HTTPException):g.queue_commit(self.commit_request(a))
    def test_non_render_columns_cannot_be_changed(self):
        a=self.claim();r=self.commit_request(a);r.changes['published_id']='overwrite'
        with self.assertRaises(HTTPException):g.queue_commit(r)
    def test_corrupt_state_not_reset(self):
        self.api.ranges.clear();before=self.api.writes
        with self.assertRaises(ClickoutError):self.fresh().initialize()
        self.assertEqual(self.api.writes,before)
    def test_failed_state_commit_has_no_queue_side_effects(self):
        a=self.claim();before=copy.deepcopy(self.api.queue)
        def concurrent():
            s=self.fresh();s.load();s.commit({'dispatch:y':{}})
        self.api.before=concurrent
        with self.assertRaises(HTTPException):g.queue_commit(self.commit_request(a))
        self.assertEqual(self.api.queue,before)

if __name__=='__main__':unittest.main()
