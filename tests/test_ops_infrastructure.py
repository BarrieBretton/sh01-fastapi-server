"""Offline contracts: authentication, bounded reads, exact B2 identity and failure handling."""
import asyncio
import os
import unittest
from unittest.mock import patch
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
import ops_infrastructure as m

ENV={'CONTROL_PLANE_API_KEY':'control-fixture','MEDIA_CLIP_API_KEY':'media-fixture',
     'BACKBLAZE_KEY_ID':'key-id','BACKBLAZE_APPLICATION_KEY':'b2-secret','BACKBLAZE_BUCKET_NAME':'owned'}
FILE={'fileId':'v1','fileName':'folder/<img>.mp4','bucketId':'bucket-1','contentLength':1234,'uploadTimestamp':1000,'action':'upload'}

class OpsTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,ENV,clear=True);self.env.start()
        app=FastAPI();app.include_router(m.router);app.include_router(m.runtime_router)
        self.client=TestClient(app);self.calls=[];self.deleted=set();self.fail_delete=False;self.bad_info=False;self.no_delete=False
        self.real_client=httpx.AsyncClient
        self.mock=patch.object(m.httpx,'AsyncClient',side_effect=lambda **kw:self.real_client(transport=httpx.MockTransport(self.provider),**kw));self.mock.start()
    def tearDown(self):
        self.mock.stop();self.env.stop();m._cache.clear()
    def provider(self,r):
        self.calls.append((r.method,r.url.path,r.content))
        p=r.url.path
        if p.endswith('b2_authorize_account'):
            return httpx.Response(200,json={'accountId':'acct','authorizationToken':'private-provider-token','apiInfo':{'storageApi':{'apiUrl':'https://api001.backblazeb2.com','bucketId':'bucket-1','bucketName':'owned','namePrefix':'','capabilities':['listFiles','readFiles']+([] if self.no_delete else ['deleteFiles'])}}})
        if p.endswith('b2_list_file_versions'):
            return httpx.Response(200,json={'files':[] if 'v1' in self.deleted else [FILE,{**FILE,'fileId':'v2','fileName':'b','contentLength':9999}], 'nextFileName':None,'nextFileId':None})
        if p.endswith('b2_get_file_info'):
            import json
            id=json.loads(r.content)['fileId']
            if id in self.deleted:return httpx.Response(404,json={'message':'private'})
            return httpx.Response(200,json={**FILE,'bucketId':'other-bucket' if self.bad_info else 'bucket-1'})
        if p.endswith('b2_delete_file_version'):
            if self.fail_delete:raise httpx.ReadTimeout('SECRET',request=r)
            self.deleted.add('v1');return httpx.Response(200,json={'fileId':'v1','fileName':FILE['fileName']})
        raise AssertionError(str(r.url))
    def get(self,path,key='control-fixture'):
        return self.client.get(path,headers={'X-API-KEY':key})
    def post(self,path,data,key='control-fixture'):
        return self.client.post('/infra/desk/b2/'+path,json=data,headers={'X-API-KEY':key})
    def row(self):
        response=self.get('/infra/desk/b2/files?slot=@default');self.assertEqual(response.status_code,200,response.text)
        return next(r for r in response.json()['rows'] if r['file_id']=='v1')
    def plan(self):
        ticket=self.row()['ticket'];r=self.post('preview',{'slot':'@default','tickets':[ticket]});self.assertEqual(r.status_code,200,r.text);return r.json()
    def test_auth_separation_no_provider_calls(self):
        for path,key in [('/infra/desk/b2/files?slot=@default','media-fixture'),('/ops/infra/inventory','control-fixture')]:
            self.assertEqual(self.get(path,key).status_code,401)
        self.assertEqual(self.calls,[])
    def test_missing_key_fail_closed(self):
        os.environ.pop('CONTROL_PLANE_API_KEY');self.assertEqual(self.get('/infra/desk/b2/files?slot=@default').status_code,503)
    def test_runtime_inventory_does_not_probe_or_expose_secrets(self):
        r=self.get('/ops/infra/inventory','media-fixture');self.assertEqual(r.status_code,200)
        self.assertNotIn('b2-secret',r.text);self.assertNotIn('control-fixture',r.text);self.assertEqual(self.calls,[])
        self.assertGreaterEqual(len(r.json()['rows']),15)
    def test_list_size_desc_exact_versions_and_no_delete(self):
        r=self.get('/infra/desk/b2/files?slot=@default');self.assertEqual(r.status_code,200,r.text)
        self.assertEqual([x['size'] for x in r.json()['rows']],[9999,1234]);self.assertIsNone(r.json()['next_cursor'])
        self.assertFalse(any(x[1].endswith('b2_delete_file_version') for x in self.calls))
        self.assertNotIn('private-provider-token',r.text)
    def test_preview_is_read_only(self):
        p=self.plan();self.assertEqual(p['bytes'],1234);self.assertEqual(p['confirmation'],'DELETE 1');self.assertFalse(self.deleted)
    def test_tampered_ticket_is_rejected(self):
        t=self.row()['ticket'];r=self.post('preview',{'slot':'@default','tickets':[t+'x']});self.assertEqual(r.status_code,409)
    def test_expired_ticket_is_rejected(self):
        t=self.row()['ticket']
        with patch.object(m.time,'time',return_value=10**12):r=self.post('preview',{'slot':'@default','tickets':[t]})
        self.assertEqual(r.status_code,409)
    def test_rotated_bucket_config_invalidates_ticket(self):
        t=self.row()['ticket'];os.environ['BACKBLAZE_BUCKET_NAME']='different';self.assertEqual(self.post('preview',{'slot':'@default','tickets':[t]}).status_code,409)
    def test_duplicate_selection_refused(self):
        t=self.row()['ticket'];self.assertEqual(self.post('preview',{'slot':'@default','tickets':[t,t]}).status_code,400)
    def test_batch_bounded(self):
        self.assertEqual(self.post('preview',{'slot':'@default','tickets':['x']*11}).status_code,422)
    def test_cannot_use_file_ticket_as_plan(self):
        t=self.row()['ticket'];self.assertEqual(self.post('delete',{'slot':'@default','plan':t,'confirmation':'DELETE 1'}).status_code,409)
    def test_confirmation_required(self):
        p=self.plan();self.assertEqual(self.post('delete',{'slot':'@default','plan':p['plan'],'confirmation':'yes'}).status_code,400);self.assertFalse(self.deleted)
    def test_cross_bucket_identity_blocks_preview(self):
        t=self.row()['ticket'];self.bad_info=True;self.assertEqual(self.post('preview',{'slot':'@default','tickets':[t]}).status_code,409)
    def test_rechecks_identity_at_execution(self):
        p=self.plan();self.bad_info=True;r=self.post('delete',{'slot':'@default','plan':p['plan'],'confirmation':p['confirmation']})
        self.assertEqual(r.json()['results'][0]['status'],'blocked');self.assertFalse(self.deleted)
    def test_success_and_replay_never_delete_newer_version(self):
        p=self.plan();body={'slot':'@default','plan':p['plan'],'confirmation':p['confirmation']}
        self.assertEqual(self.post('delete',body).json()['results'][0]['status'],'deleted')
        self.assertEqual(self.post('delete',body).json()['results'][0]['status'],'blocked')
        self.assertEqual(sum(c[1].endswith('b2_delete_file_version') for c in self.calls),1)
    def test_timeout_is_unknown_and_never_retried(self):
        p=self.plan();self.fail_delete=True;r=self.post('delete',{'slot':'@default','plan':p['plan'],'confirmation':p['confirmation']})
        self.assertEqual(r.json()['results'][0]['status'],'unknown');self.assertNotIn('SECRET',r.text)
        self.assertEqual(sum(c[1].endswith('b2_delete_file_version') for c in self.calls),1)
    def test_missing_delete_capability(self):
        self.no_delete=True;t=self.row()['ticket'];self.assertEqual(self.post('preview',{'slot':'@default','tickets':[t]}).status_code,403)
    def test_runtime_cannot_delete(self):
        self.assertEqual(self.client.post('/ops/infra/b2/delete',json={},headers={'X-API-KEY':'media-fixture'}).status_code,404)
    def test_resource_limits_are_measured_not_assumed_512mb(self):
        rows=m.resource_metrics();self.assertTrue(rows);self.assertTrue(all(r['used']>=0 for r in rows))
    def test_budget_unknown_or_operator_defined(self):
        self.assertIsNone(m.budget('b2:@default'))
        os.environ['OPS_INFRA_LIMITS_JSON']='{"b2:@default":{"storage_bytes":100000}}';self.assertEqual(m.budget('b2:@default'),100000)
    def test_disallowed_monitor_origin(self):
        for value in ['http://x','https://user:pass@x','https://x?token=s']:
            with self.assertRaises(m.HTTPException):m.origin(value)

if __name__=='__main__':unittest.main()

class StorageUsageTests(unittest.IsolatedAsyncioTestCase):
    def bucket(self):
        return m.B2(None,'https://api001.backblazeb2.com','private','bucket','name','folder/',['listFiles'],'@default','scope')

    async def test_pagination_counts_uploaded_versions_once_and_preserves_prefix(self):
        b=self.bucket();calls=[]
        async def call(method,args):
            calls.append(dict(args))
            if len(calls)==1:return {'files':[FILE,{**FILE,'fileId':'hide','action':'hide','contentLength':99}], 'nextFileName':'folder/next','nextFileId':'v2'}
            return {'files':[FILE,{**FILE,'fileId':'v2','contentLength':200}], 'nextFileName':None}
        b.call=call
        d=await b.storage_usage()
        self.assertEqual(d['bytes'],1434);self.assertEqual(d['versions'],2);self.assertTrue(d['complete'])
        self.assertEqual(d['scope'],'prefix');self.assertEqual(calls[1]['prefix'],'folder/')
        self.assertEqual(calls[1]['startFileId'],'v2')

    async def test_scan_is_bounded_and_partial(self):
        b=self.bucket();calls=[]
        async def call(method,args):
            calls.append(dict(args));n=len(calls)
            return {'files':[{**FILE,'fileId':str(n)}],'nextFileName':str(n),'nextFileId':str(n)}
        b.call=call;d=await b.storage_usage()
        self.assertEqual(len(calls),10);self.assertFalse(d['complete']);self.assertEqual(d['versions'],10)

    async def test_later_timeout_keeps_partial_usage(self):
        b=self.bucket();calls=[]
        async def call(method,args):
            calls.append(1)
            if len(calls)>1:raise httpx.ReadTimeout('private')
            return {'files':[FILE],'nextFileName':'next','nextFileId':'next'}
        b.call=call;d=await b.storage_usage()
        self.assertFalse(d['complete']);self.assertEqual(d['bytes'],1234)

    async def test_media_card_returns_storage_without_control_key(self):
        from unittest.mock import AsyncMock
        with patch.dict(os.environ,ENV,clear=True):
            os.environ.pop('CONTROL_PLANE_API_KEY')
            b=self.bucket()
            b.storage_usage=AsyncMock(return_value={'bytes':1234,'versions':1,'pages':1,'complete':True,'scope':'prefix','prefix':'folder/','observed_at':'now'})
            with patch.object(m.B2,'connect',new=AsyncMock(return_value=b)):
                item=await m.probe('b2:@default',True)
            self.assertEqual(item['status'],'healthy');self.assertEqual(item['metrics'][0]['used'],1234)
            self.assertIsNone(item['metrics'][0]['limit']);self.assertEqual(item['bucket_id'],'bucket')
