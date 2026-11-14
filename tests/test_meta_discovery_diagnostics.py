import json
import os
import unittest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import social_inventory as inv
import social_meta_accounts as meta

class DiscoveryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MEDIA_CLIP_API_KEY':'test-key', 'FB_PAGE_ACCESS_TOKEN':'x'*32}, clear=True)
        self.env.start(); meta._cache.clear()
    def tearDown(self):
        self.env.stop(); meta._cache.clear()
    def test_http_error_retains_only_numeric_codes_and_fixed_hints(self):
        response = Mock(status=400)
        response.read.return_value = json.dumps({'error':{'code':100, 'error_subcode':33, 'type':'OAuthException',
            'message':'Nonexisting field (accounts) on node type (Page). SECRET_VALUE https://private.invalid',
            'fbtrace_id':'DO_NOT_RETURN'}}).encode()
        conn = Mock(); conn.getresponse.return_value = response
        with patch.object(inv, 'validate_url', return_value=('graph.facebook.com','1.1.1.1','/v24.0/me/accounts')), patch.object(inv,'PinnedHTTPS',return_value=conn):
            with self.assertRaises(inv.GraphHTTPError) as caught:
                inv.SafeHTTP(inv.Budget()).json('https://graph.facebook.com/v24.0/me/accounts',('graph.facebook.com',))
        self.assertEqual(caught.exception.diagnostic, {'http_status':400,'graph_code':100,'graph_subcode':33,
            'graph_type':'OAuthException','hint':'request_resolved_to_page_node'})
        response.read.assert_called_once_with(16385)
        conn.close.assert_called_once()
    def test_failed_edges_reach_authenticated_response_without_raw_body(self):
        class Failed:
            def json(self, *args, **kwargs):
                raise inv.GraphHTTPError('upstream_http_error',400,{'error':{'code':190,'message':'SECRET_VALUE'}})
        with self.assertRaises(meta.DiscoveryError) as caught:
            meta.discover(Failed())
        errors = caught.exception.errors
        self.assertEqual([e['edge'] for e in errors],['me/accounts','me/assigned_pages'])
        self.assertTrue(all(e['graph_code']==190 for e in errors))
        self.assertNotIn('SECRET_VALUE',str(errors))
    def test_transport_errors_are_reported_per_edge(self):
        class Failed:
            def json(self, *args, **kwargs): raise inv.SafeError('upstream_transport_error')
        with self.assertRaises(meta.DiscoveryError) as caught:meta.discover(Failed())
        self.assertEqual(len(caught.exception.errors),2)
        self.assertEqual(caught.exception.errors[0]['code'],'upstream_transport_error')
    def test_oversized_or_non_numeric_fields_are_not_exposed(self):
        e=inv.GraphHTTPError('upstream_http_error',400,{'error':{'code':'SECRET_VALUE','error_subcode':True,'type':'SECRET_VALUE'}})
        self.assertEqual(e.diagnostic,{'http_status':400})
        response=Mock(status=500);response.read.return_value=b'x'*16385
        conn=Mock();conn.getresponse.return_value=response
        with patch.object(inv,'validate_url',return_value=('graph.facebook.com','1.1.1.1','/')),patch.object(inv,'PinnedHTTPS',return_value=conn):
            with self.assertRaises(inv.GraphHTTPError) as caught:
                inv.SafeHTTP(inv.Budget()).json('https://graph.facebook.com/',('graph.facebook.com',))
        self.assertEqual(caught.exception.diagnostic,{'http_status':500})

if __name__=='__main__':unittest.main()
