import unittest
from unittest.mock import Mock, patch
import social_inventory as core
import scanner_quota as q

class QuotaTests(unittest.TestCase):
    def setUp(self):
        q._STATES.clear()
        self.http=Mock();self.http.budget.remaining.return_value=300
    def test_rate_limit_shared_by_reads_and_writes_without_extra_requests(self):
        e=core.SafeError('upstream_http_error');e.diagnostic={'http_status':503,'dashboard_code':'store_rate_limited'}
        self.http.json.side_effect=e
        a=q.QuotaHTTP(self.http,'one')
        with self.assertRaises(core.SafeError):a.json('url')
        b=q.QuotaHTTP(self.http,'one')
        with self.assertRaises(core.SafeError) as caught:b.json('url',method='POST')
        self.assertGreaterEqual(caught.exception.diagnostic['retry_after_seconds'],75)
        self.assertEqual(self.http.json.call_count,1)
    def test_scope_isolates_credentials_and_expiry_recovers(self):
        a=q.QuotaHTTP(self.http,'one');a.state['cooldown']=100
        with patch.object(q.time,'monotonic',return_value=101):a.json('url')
        self.assertEqual(self.http.json.call_count,1)
        self.assertIsNot(a.state,q.QuotaHTTP(self.http,'two').state)
        self.assertNotEqual(q.scope('x',{'key':'a'}),q.scope('x',{'key':'b'}))
    def test_permanent_errors_not_retried(self):
        e=core.SafeError('upstream_http_error');e.diagnostic={'http_status':503,'dashboard_code':'store_google_permission_denied'}
        self.http.json.side_effect=e
        with self.assertRaises(core.SafeError):q.QuotaHTTP(self.http,'one').json('url')
        self.assertEqual(self.http.json.call_count,1)
    def test_read_transient_retries_bounded(self):
        e=core.SafeError('upstream_http_error');e.diagnostic={'http_status':502}
        self.http.json.side_effect=e
        with patch.object(q.time,'sleep'), self.assertRaises(core.SafeError):q.QuotaHTTP(self.http,'one').json('url')
        self.assertEqual(self.http.json.call_count,3)
    def test_pacing_respects_budget(self):
        a=q.QuotaHTTP(self.http,'one');a.state['next']=q.time.monotonic()+10
        self.http.budget.remaining.return_value=2
        with self.assertRaises(core.SafeError):a.json('url')
        self.http.json.assert_not_called()

class ClientRecoveryTests(unittest.TestCase):
    def test_runner_retries_same_request_and_stops_on_permanent_error(self):
        import io, json
        import run_social_scan as runner
        opener=Mock()
        throttle={'ok':False,'errors':[{'diagnostic':{'dashboard_code':'store_rate_limited','retry_after_seconds':90}}]}
        opener.open.side_effect=[io.BytesIO(json.dumps(throttle).encode()),io.BytesIO(b'{"ok":true}')]
        req=object()
        with patch.object(runner.time,'sleep') as sleep:
            self.assertTrue(runner.request_batch(opener,req)['ok'])
        self.assertGreaterEqual(sleep.call_args.args[0],90)
        self.assertIs(opener.open.call_args_list[0].args[0],opener.open.call_args_list[1].args[0])
        self.assertIsNone(runner.throttled_delay({'errors':[{'diagnostic':{'dashboard_code':'invalid_fingerprint'}}]}))
    def test_catalogue_failure_retains_original_cursor(self):
        import dashboard_scanner as ds
        request=core.ScanRequest(platform='facebook',account_id='1',max_posts=2,cursor='opaque')
        result={'ok':False,'resume_cursor':'advanced','counts':{'posts_ingested':0},'coverage':{'provider_exhausted':True},'errors':[{'diagnostic':{'retry_after_seconds':75}}]}
        with patch.object(ds,'bridge_client'),patch.object(ds,'FingerprintSink'),patch.object(core,'scan',return_value=result):
            out=ds.scan_catalogue(request)
        self.assertEqual(out['resume_cursor'],'opaque')
        self.assertFalse(out['coverage']['catalogue_exhausted'])
        self.assertEqual(out['retry_after_seconds'],75)
