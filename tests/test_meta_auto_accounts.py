import os
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit, parse_qs
import social_inventory as inv
import social_meta_accounts as meta

TOKEN = 'x' * 32
PAGE = 'p' * 32

class FakeHTTP:
    def __init__(self): self.calls = []
    def json(self, url, allowed, **kwargs):
        self.calls.append((url, kwargs))
        if 'assigned_pages' in url:
            return {'data': [{'id': '12', 'name': 'Other'}]}
        if parse_qs(urlsplit(url).query).get('after'):
            return {'data': [{'id': '11', 'name': 'Second'}]}
        return {'data': [{'id': '10', 'name': 'First', 'access_token': PAGE,
                          'instagram_business_account': {'id': '99'}}],
                'paging': {'next': 'https://evil.invalid/?access_token=DO_NOT_FOLLOW', 'cursors': {'after': 'next'}}}

class AutoAccountsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'META_SYSVAR': TOKEN, 'MEDIA_CLIP_API_KEY': 'media'}, clear=True)
        self.env.start(); meta._cache.clear()
    def tearDown(self): self.env.stop(); meta._cache.clear()
    def test_existing_fb_key_fallback_and_explicit_precedence(self):
        os.environ.pop('META_SYSVAR')
        os.environ['FB_PAGE_ACCESS_TOKEN'] = TOKEN
        self.assertEqual(meta.credential(), TOKEN)
        self.assertEqual(meta.snapshot()['meta_credential_source'], 'FB_PAGE_ACCESS_TOKEN')
        meta.discover(FakeHTTP())
        os.environ['META_SYSVAR'] = 'z' * 32
        self.assertEqual(meta.credential(), 'z' * 32)
        self.assertEqual(meta.snapshot()['discovered_accounts'], 0)
        os.environ['META_SYSVAR'] = 'invalid'
        with self.assertRaises(inv.SafeError): meta.credential()
    def test_discovery_pagination_dedup_cache_and_redaction(self):
        http = FakeHTTP()
        accounts, coverage = meta.discover(http)
        self.assertEqual(len(accounts), 4)
        self.assertEqual(accounts[('instagram', '99')]['_meta_token'], PAGE)
        self.assertEqual(len(http.calls), 3)
        meta.discover(http); self.assertEqual(len(http.calls), 3)
        self.assertNotIn(PAGE, str(meta.public_accounts()))
        self.assertNotIn(TOKEN, str(meta.snapshot()))
        self.assertFalse(coverage['all_accessible_accounts_verified'])
        self.assertTrue(all('evil.invalid' not in c[0] for c in http.calls))
    def test_json_token_and_rotation(self):
        os.environ['META_SYSVAR'] = '{"access_token":"' + TOKEN + '"}'
        meta.discover(FakeHTTP())
        os.environ['META_SYSVAR'] = 'z' * 32
        self.assertEqual(meta.snapshot()['discovered_accounts'], 0)
        http = FakeHTTP(); meta.discover(http); self.assertEqual(len(http.calls), 3)
    def test_config_lookup_and_provider_uses_page_token(self):
        meta.discover(FakeHTTP())
        cfg = inv.config_for('99', 'instagram')
        http = FakeHTTP(); inv.Provider(cfg, http).get('v24.0/99/media', {})
        self.assertEqual(http.calls[0][1]['headers']['Authorization'], 'Bearer ' + PAGE)
        with self.assertRaises(inv.SafeError): inv.config_for('777', 'instagram')
    def test_partial_error_is_visible(self):
        class Partial(FakeHTTP):
            def json(self, url, allowed, **kwargs):
                if 'assigned_pages' in url: raise inv.SafeError('upstream_permission_denied')
                return super().json(url, allowed, **kwargs)
        accounts, coverage = meta.discover(Partial())
        self.assertEqual(len(accounts), 3); self.assertEqual(len(coverage['errors']), 1)
    def test_invalid_pagination_not_claimed_complete(self):
        class Broken:
            def json(self, *args, **kwargs): return {'data': [], 'paging': {'next': 'untrusted'}}
        with self.assertRaises(inv.SafeError): meta.discover(Broken())
    def test_default_dashboard_reuses_media_key_only_for_fixed_origin(self):
        self.assertEqual(inv.Dashboard(None).base, inv.DEFAULT_DASHBOARD_URL)
        self.assertEqual(inv.dashboard_ingest_key(), 'media')
        os.environ['OPS_DASHBOARD_URL'] = 'https://different.example'
        with self.assertRaises(inv.SafeError): inv.dashboard_ingest_key()
        os.environ['OPS_INGEST_TOKEN'] = 'explicit'
        self.assertEqual(inv.dashboard_ingest_key(), 'explicit')
    def test_failed_cache_prevents_provider_media_work(self):
        from unittest.mock import Mock
        dashboard = Mock()
        dashboard.cache.side_effect = inv.SafeError('upstream_permission_denied')
        provider = Mock()
        with self.assertRaisesRegex(inv.SafeError, 'dashboard_cache_unavailable'):
            inv.scan(inv.ScanRequest(platform='facebook', account_id='10', max_posts=1),
                     cfg={'platform':'facebook','account_id':'10'}, provider=provider, dashboard=dashboard)
        provider.page.assert_not_called()
        dashboard.ingest.assert_not_called()
    def test_missing_meta_status_is_honest_and_accounts_auth_required(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(inv.router); client = TestClient(app)
        self.assertEqual(client.get('/social/inventory/accounts').status_code, 401)
        os.environ.pop('META_SYSVAR')
        r = client.get('/social/inventory/status', headers={'X-API-KEY': 'media'}).json()
        self.assertFalse(r['meta_credential_configured'])
        self.assertFalse(r['dashboard_auth_verified'])
        self.assertEqual(client.get('/social/inventory/accounts', headers={'X-API-KEY': 'media'}).status_code, 503)

if __name__ == '__main__': unittest.main()
