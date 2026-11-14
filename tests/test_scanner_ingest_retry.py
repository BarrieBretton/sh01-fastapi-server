import json
import os
import unittest
from unittest.mock import Mock, patch
import social_inventory as core
import dashboard_scanner as ds

ROW={'platform':'facebook','account_id':'1','post_id':'2','fingerprint':{'algorithm':'sh01-social-perceptual','version':1,'media_type':'IMAGE'}}

class ScannerWriteTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'MEDIA_CLIP_API_KEY':'private-key'},clear=True);self.env.start()
        self.http=Mock();self.http.budget=core.Budget(seconds=30);self.sink=ds.FingerprintSink(self.http)
    def tearDown(self):self.env.stop()
    def failure(self,code='store_rate_limited',status=503,header=None):
        return core.DashboardHTTPError('upstream_http_error',status,{'error':code,'message':'SECRET'},header)
    def test_rate_limit_retries_identical_batch_with_jitter_and_retry_after(self):
        self.http.json.side_effect=[self.failure(header='3'),{'accepted':1}]
        with patch.object(ds.random,'uniform',return_value=.5),patch.object(ds.time,'sleep') as sleep:
            self.assertEqual(self.sink.ingest('posts',[ROW]),1)
        sleep.assert_called_once_with(3.5)
        self.assertEqual(self.http.json.call_args_list[0],self.http.json.call_args_list[1])
    def test_permanent_validation_error_never_retried_and_diagnostics_survive(self):
        self.http.json.side_effect=self.failure('scanner_media_type_mismatch',400)
        with patch.object(ds.time,'sleep') as sleep:
            with self.assertRaises(core.IngestError) as caught:self.sink.ingest('posts',[ROW])
        self.assertEqual(self.http.json.call_count,1);sleep.assert_not_called()
        self.assertEqual(caught.exception.diagnostic['dashboard_code'],'scanner_media_type_mismatch')
        self.assertEqual(caught.exception.diagnostic['http_status'],400)
        self.assertNotIn('SECRET',str(caught.exception.diagnostic))
    def test_storage_configuration_failure_not_retried_even_when_503(self):
        self.http.json.side_effect=self.failure('store_google_scope_missing')
        with self.assertRaises(core.IngestError):self.sink.ingest('posts',[ROW])
        self.assertEqual(self.http.json.call_count,1)
    def test_maximum_three_attempts_and_partial_acceptance_is_preserved(self):
        rows=[dict(ROW,post_id=str(i)) for i in range(11)]
        self.http.json.side_effect=[{'accepted':10},self.failure(),self.failure(),self.failure()]
        with patch.object(ds.time,'sleep'):
            with self.assertRaises(core.IngestError) as caught:self.sink.ingest('posts',rows)
        self.assertEqual(caught.exception.accepted,10)
        self.assertEqual(caught.exception.diagnostic['write_attempts'],3)
        self.assertEqual(self.http.json.call_count,4)
    def test_long_retry_after_keeps_original_diagnostic_and_does_not_sleep(self):
        self.http.json.side_effect=self.failure(header='120')
        with patch.object(ds.time,'sleep') as sleep:
            with self.assertRaises(core.IngestError) as caught:self.sink.ingest('posts',[ROW])
        sleep.assert_not_called();self.assertEqual(self.http.json.call_count,1)
        self.assertEqual(caught.exception.diagnostic['retry_deferred'],'delivery_budget_insufficient')
        self.assertEqual(caught.exception.diagnostic['retry_after_seconds'],120)
    def test_ambiguous_transport_failure_can_replay_deterministic_group_upsert(self):
        self.http.json.side_effect=[core.TransportError(ConnectionResetError(104,'private text'),'response_headers'),{'accepted':1}]
        group={'group_id':'dup_same_id','members':['1','2']}
        with patch.object(ds.time,'sleep'):
            self.assertEqual(self.sink.ingest('groups',[group]),1)
        self.assertEqual(self.http.json.call_args_list[0],self.http.json.call_args_list[1])
    def test_transport_extracts_only_known_dashboard_error_fields(self):
        response=Mock(status=503);response.read.return_value=json.dumps({'error':'store_rate_limited','secret':'private-body'}).encode()
        response.getheader.return_value='2'
        conn=Mock();conn.getresponse.return_value=response
        host='sh01-operations-dashboard-cloudflare.vivojaymail.workers.dev'
        with patch.object(core,'validate_url',return_value=(host,['1.1.1.1'],'/api/ingest/scanner/fingerprints')),patch.object(core,'PinnedHTTPS',return_value=conn):
            with self.assertRaises(core.DashboardHTTPError) as caught:
                core.SafeHTTP(core.Budget()).json(core.DEFAULT_DASHBOARD_URL+'/api/ingest/scanner/fingerprints',(host,))
        self.assertEqual(caught.exception.diagnostic,{'http_status':503,'dashboard_code':'store_rate_limited','retry_after_seconds':2})
        response.read.assert_called_once_with(16385)
        self.assertNotIn('private-body',str(caught.exception.diagnostic))
        unknown=self.failure('PRIVATE_VALUE');self.assertNotIn('dashboard_code',unknown.diagnostic)
    def test_scan_returns_save_diagnostic_and_rewinds_unacknowledged_batch(self):
        dash=Mock();dash.cache.return_value=({'2':{'media_type':'IMAGE'}},False)
        dash.ingest.side_effect=core.IngestError('upstream_http_error',0,'posts',{'http_status':503,'dashboard_code':'store_rate_limited','write_attempts':3})
        provider=Mock(spec=['page','media_url']);provider.page.return_value=([{'id':'2'}],None)
        metadata={'platform':'facebook','account_id':'1','post_id':'2','media_type':'IMAGE'}
        with patch.object(core,'valid_fingerprint',return_value=True):
            result=core.scan(core.ScanRequest(platform='facebook',account_id='1',max_posts=1),
                             cfg={'platform':'facebook','account_id':'1'},provider=provider,dashboard=dash,
                             normalizer=lambda *args:core.Post(metadata))
        self.assertFalse(result['ok'])
        self.assertEqual(result['errors'][0]['diagnostic']['dashboard_code'],'store_rate_limited')
        self.assertEqual(result['coverage']['stop_reason'],'dashboard_ingest_failed_replay')
        self.assertEqual(core.decode_cursor(result['resume_cursor'],'1','facebook'),{'after':None,'skip':0})
        provider.media_url.assert_not_called()

if __name__=='__main__':unittest.main()
