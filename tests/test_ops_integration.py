import ast,asyncio,json,logging,os,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import social_inventory as inventory
import sh01_error_handler as telemetry
class IntegrationTests(unittest.TestCase):
 def test_status_auth_and_config_shape(self):
  app=FastAPI();app.include_router(inventory.router);client=TestClient(app)
  with patch.dict(os.environ,{'SOCIAL_INVENTORY_API_KEY':'test-key','SOCIAL_ACCOUNTS_JSON':json.dumps({'123':{'platform':'instagram','handle':'example','token_env':'PRIVATE_ENV'}})},clear=True):
   self.assertEqual(client.get('/social/inventory/status').status_code,401)
   r=client.get('/social/inventory/status',headers={'X-API-KEY':'test-key'})
   self.assertEqual(r.status_code,200);self.assertEqual(r.json()['configured_accounts'],1);self.assertFalse(r.json()['delete_enabled']);self.assertNotIn('PRIVATE_ENV',r.text)
 def test_error_payload_excludes_messages_and_args(self):
  with patch.object(telemetry.threading.Thread,'start'),patch.dict(os.environ,{'OPS_DASHBOARD_URL':'https://dashboard.test','OPS_INGEST_TOKEN':'secret'}):
   h=telemetry.OpsHandler();record=logging.LogRecord('test',logging.ERROR,'test.py',1,'Bearer raw-secret %s',('argument-secret',),None);h.emit(record);payload=json.dumps(h.events.get_nowait());self.assertNotIn('raw-secret',payload);self.assertNotIn('argument-secret',payload);self.assertIn('ERROR',payload)
 def test_existing_app_mounts_both_routes_and_root_telemetry(self):
  source=(Path(__file__).resolve().parents[1]/'app.py').read_text();ast.parse(source)
  self.assertIn('app.include_router(social_inventory_router)',source);self.assertIn('app.include_router(social_delete_router)',source);self.assertIn('install_ops_handler(logging.getLogger())',source)
if __name__=='__main__':unittest.main()
