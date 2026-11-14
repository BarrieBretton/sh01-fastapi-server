import os, unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import social_inventory as inventory
import social_delete as deletion

class KeyFallbackTests(unittest.TestCase):
    def setUp(self):
        app=FastAPI();app.include_router(inventory.router);app.include_router(deletion.router)
        self.client=TestClient(app)
        self.env=patch.dict(os.environ,{},clear=True);self.env.start()
    def tearDown(self):self.env.stop()
    def test_missing_keys_reject(self):
        self.assertEqual(self.client.get('/social/inventory/status').status_code,503)
        with self.assertRaises(inventory.SafeError):inventory.inventory_cursor_key()
    def test_media_key_authenticates_status_and_scan(self):
        os.environ['MEDIA_CLIP_API_KEY']='media-test'
        headers={'X-API-KEY':'media-test'}
        self.assertEqual(self.client.get('/social/inventory/status').status_code,401)
        response=self.client.get('/social/inventory/status',headers=headers)
        self.assertEqual(response.status_code,200);self.assertNotIn('media-test',response.text)
        with patch('dashboard_scanner.scan_catalogue',return_value={'ok':True}) as scan:
            r=self.client.post('/social/inventory/scan',headers=headers,json={'platform':'instagram','account_id':'123','max_posts':1})
            self.assertEqual(r.status_code,200);scan.assert_called_once()
    def test_explicit_override_takes_precedence(self):
        os.environ.update(MEDIA_CLIP_API_KEY='media-test',SOCIAL_INVENTORY_API_KEY='override-test')
        self.assertEqual(self.client.get('/social/inventory/status',headers={'X-API-KEY':'media-test'}).status_code,401)
        self.assertEqual(self.client.get('/social/inventory/status',headers={'X-API-KEY':'override-test'}).status_code,200)
    def test_cursor_signing_uses_selected_key_and_rotation_invalidates(self):
        os.environ['MEDIA_CLIP_API_KEY']='media-test'
        cursor=inventory.encode_cursor({'after':'next','skip':0},'123','instagram')
        self.assertEqual(inventory.decode_cursor(cursor,'123','instagram')['after'],'next')
        os.environ['SOCIAL_INVENTORY_API_KEY']='override-test'
        with self.assertRaises(inventory.SafeError):inventory.decode_cursor(cursor,'123','instagram')
    def test_media_key_does_not_enable_deletion(self):
        os.environ['MEDIA_CLIP_API_KEY']='media-test'
        import asyncio
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as e:asyncio.run(deletion.delete_route(None,x_api_key='media-test'))
        self.assertEqual(e.exception.status_code,503)
    def test_ingest_alias_and_status_agree(self):
        os.environ.update(MEDIA_CLIP_API_KEY='media-test',OPS_DASHBOARD_URL='https://dashboard.test',OPS_INGEST_TOKEN='ops-test')
        self.assertEqual(inventory.Dashboard(None).headers['X-INGEST-KEY'],'ops-test')
        os.environ['INGEST_TOKEN']='ingest-test'
        self.assertEqual(inventory.Dashboard(None).headers['X-INGEST-KEY'],'ingest-test')
        r=self.client.get('/social/inventory/status',headers={'X-API-KEY':'media-test'})
        self.assertTrue(r.json()['dashboard_configured']);self.assertNotIn('ingest-test',r.text)
