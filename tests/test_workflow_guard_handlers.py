import copy,json,subprocess,unittest,shutil
from pathlib import Path
from scripts.sh01_workflow_transform import *
@unittest.skipUnless(shutil.which('node'), 'Node is optional; JS tests require Node')
class TransformTests(unittest.TestCase):
 def setUp(self):
  self.w={SCHEDULER:{'nodes':[], 'connections':{}, 'settings':{'timezone':'Asia/Kolkata'}},TELEGRAM:{'nodes':[], 'connections':{},'settings':{'timezone':'Asia/Kolkata'}}}
  names={SCHEDULER:['filter','n8n-workflows-data','get-only-the-correct-workflow','gen-payload','delegate-all-tasks'],TELEGRAM:['Select Queue Work','Poll Render Job','Submit Render Job','Handle Poll Response','Handle Submit Response','Update Render Queue State','Route Queue Action']}
  for wid, entries in names.items():
   for name in entries:
    self.w[wid]['nodes'].append({'name':name,'parameters':{'jsCode':"return [{json:{action:'idle'}}];"}})
  n=next(n for n in self.w[TELEGRAM]['nodes'] if n['name']=='Submit Render Job')
  n['credentials']={'httpHeaderAuth':{'id':'fixture-ref','name':'fixture'}}
 def js(self,code,input,refs):
  script='const [code,input,refs]=JSON.parse(process.argv[1]); const fn=new Function("$input","$",code); const result=fn({all:()=>input,first:()=>input[0]},n=>({all:()=>refs[n],first:()=>refs[n][0]})); console.log(JSON.stringify(result));'
  return json.loads(subprocess.check_output(['node','-e',script,json.dumps([code,input,refs])],text=True,stderr=subprocess.PIPE))
 def test_distinct_targets_keep_their_rows(self):
  refs={'filter':[{'json':{'workflow_name':'A','row_number':2}},{'json':{'workflow_name':'B','row_number':3}}]}
  result=self.js(RESOLVE,[{'json':{'id':'a','name':'A'}},{'json':{'id':'b','name':'B'}}],refs)
  self.assertEqual([(x['json']['workflow_id'],x['json']['row_number']) for x in result],[('a',2),('b',3)])
 def test_same_target_different_rows_both_survive(self):
  refs={'filter':[{'json':{'workflow_name':'A','row_number':2}},{'json':{'workflow_name':'A','row_number':3}}]}
  self.assertEqual(len(self.js(RESOLVE,[{'json':{'id':'a','name':'A'}}],refs)),2)
 def test_missing_and_ambiguous_targets_stop_before_claims(self):
  for rows in [[],[{'json':{'id':'a','name':'A'}},{'json':{'id':'b','name':'A'}}]]:
   with self.assertRaises(subprocess.CalledProcessError):self.js(RESOLVE,rows,{'filter':[{'json':{'workflow_name':'A'}}]})
 def test_ist_and_server_rows(self):
  rows=[{'json':{'row_number':2,'workflow':'A','timings':'11:15pm'}},{'json':{'row_number':3,'workflow':'Server::Snapchat','timings':'11:15pm'}}]
  result=self.js(FILTER,rows,{'Schedule Trigger':[{'json':{'timestamp':'2026-11-18T17:45:00Z'}}]})
  self.assertEqual([r['json']['workflow_name'] for r in result],['A'])
 def test_queue_states_and_transient_errors_preserve_identity(self):
  q={'queue_id':'q','status':'rendering','render_job_id':'av-q','attempt_count':2}
  refs={'Unwrap Queue Claim':[{'json':q}]}
  for status in ['queued','accepted','running']:
   out=self.js(HANDLER,[{'json':{'statusCode':202,'body':{'status':status,'job_id':'av-q'}}}],refs)[0]['json']
   self.assertEqual((out['status'],out['attempt_count'],out['render_job_id']),('rendering',2,'av-q'))
  for status in [404,409,429,500,503]:
   out=self.js(HANDLER,[{'json':{'statusCode':status,'body':{}}}],refs)[0]['json']
   self.assertEqual((out['status'],out['attempt_count'],out['render_job_id']),('rendering',2,'av-q'))
 def test_submission_queued_records_job(self):
  refs={'Unwrap Queue Claim':[{'json':{'queue_id':'q','status':'staged','attempt_count':0}}]}
  out=self.js(HANDLER,[{'json':{'statusCode':202,'body':{'status':'queued','job_id':'av-q'}}}],refs)[0]['json']
  self.assertEqual((out['status'],out['attempt_count'],out['render_job_id']),('rendering',0,'av-q'))
 def test_confirmed_failure_is_not_auto_retried(self):
  refs={'Unwrap Queue Claim':[{'json':{'queue_id':'q','status':'rendering','attempt_count':0,'render_job_id':'av-q'}}]}
  out=self.js(HANDLER,[{'json':{'statusCode':200,'body':{'status':'failed'}}}],refs)[0]['json']
  self.assertEqual(out['status'],'failed');self.assertEqual(out['render_job_id'],'av-q')
 def test_edges_and_credentials_preserved(self):
  before=copy.deepcopy(self.w);out=transform(self.w,'https://guard.example')
  self.assertEqual(self.w,before)
  for w in out.values():
   names={n['name'] for n in w['nodes']}
   for source,connections in w['connections'].items():
    self.assertIn(source,names)
    for groups in connections.values():
     for group in groups:
      for edge in group:self.assertIn(edge['node'],names)
  node=next(n for n in out[TELEGRAM]['nodes'] if n['name']=='Update Render Queue State')
  self.assertEqual(node['type'],'n8n-nodes-base.httpRequest')
  self.assertEqual(node['credentials']['httpHeaderAuth']['id'],'fixture-ref')
  self.assertEqual(out[SCHEDULER]['settings'],before[SCHEDULER]['settings'])
 def test_denied_claim_never_dispatches(self):
  out=self.js(UNWRAP_DISPATCH,[{'json':{'allowed':False}},{'json':{'allowed':True,'payload':{'id':'ok'}}}],{})
  self.assertEqual(out,[{'json':{'id':'ok'}}])

if __name__=='__main__':unittest.main()
