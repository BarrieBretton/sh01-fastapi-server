import os,sys,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import social_delete as d
class HTTP:
 def json(self,*a,**kw):return {'post':{'platform':'instagram','account_id':'100','post_id':'200','permalink':'https://www.instagram.com/p/ABC/','delete_status':'published'}}
class Dashboard:
 base='https://dashboard.test';hosts=('dashboard.test',);headers={};http=HTTP()
class Provider:
 def __init__(self,cfg,http):self.cfg=cfg;self.version='v24.0';self.http=http
 def get(self,*a,**kw):return {'id':'200','owner':{'id':'100'},'permalink':'https://www.instagram.com/p/ABC/'}
class CleanupTests(unittest.TestCase):
 def setUp(self):self.config=patch.dict(os.environ,{'SOCIAL_ACCOUNTS_JSON':'{"100":{"platform":"instagram","handle":"example","token_env":"TEST_TOKEN","allow_delete":true}}'},clear=True);self.config.start()
 def tearDown(self):self.config.stop()
 def req(self,**kw):return d.DeleteRequest(platform='instagram',permalink='https://www.instagram.com/p/ABC/',**kw)
 def test_dry_run_never_deletes(self):
  with patch.object(d,'perform_delete') as delete:
   result=d.delete_post(self.req(),dashboard=Dashboard(),provider_factory=Provider);self.assertEqual(result['status'],'preview');delete.assert_not_called()
 def test_config_and_confirmation_required(self):
  with self.assertRaisesRegex(d.SafeError,'not_enabled'):d.delete_post(self.req(dry_run=False),dashboard=Dashboard(),provider_factory=Provider)
  os.environ['SOCIAL_DELETE_ENABLED']='1'
  with self.assertRaisesRegex(d.SafeError,'confirmation'):d.delete_post(self.req(dry_run=False,confirmation='yes'),dashboard=Dashboard(),provider_factory=Provider)
 def test_exact_confirm_and_owner_check(self):
  os.environ['SOCIAL_DELETE_ENABLED']='1'
  with patch.object(d,'perform_delete') as delete:
   r=d.delete_post(self.req(dry_run=False,confirmation='DELETE instagram:100:200'),dashboard=Dashboard(),provider_factory=Provider);self.assertEqual(r['status'],'deleted');delete.assert_called_once()
  class Wrong(Provider):
   def get(self,*a,**kw):return {'owner':{'id':'999'},'permalink':'https://www.instagram.com/p/ABC/'}
  with patch.object(d,'perform_delete') as delete:
   with self.assertRaisesRegex(d.SafeError,'owner'):d.delete_post(self.req(dry_run=False,confirmation='DELETE instagram:100:200'),dashboard=Dashboard(),provider_factory=Wrong)
   delete.assert_not_called()
if __name__=='__main__':unittest.main()
