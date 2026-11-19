"""Pure transforms: preserve credential references from live workflow reads."""
import copy, json, re, uuid
REVISION='workflow-guards-1'
SCHEDULER='cSoy04ojAzkzBkVM'
TELEGRAM='2R3VrBB58stPVz6q'
FILTER=r'''const trigger = $('Schedule Trigger').first().json;
const timestamp = trigger.timestamp;
const instant = new Date(timestamp);
if (!timestamp || !Number.isFinite(instant.getTime())) throw new Error('Missing schedule timestamp');
const parts = new Intl.DateTimeFormat('en-GB', {timeZone:'Asia/Kolkata', hour:'2-digit', minute:'2-digit', hourCycle:'h23'}).formatToParts(instant);
const minute = Number(parts.find(x=>x.type==='hour').value)*60 + Number(parts.find(x=>x.type==='minute').value);
function parse(v) {
 const m=String(v).trim().toLowerCase().replace(/\s+/g,'').match(/^(\d{1,2})(?::(\d{2}))?(am|pm)$/);
 if (!m || +m[1]<1 || +m[1]>12 || +(m[2]||0)>59) throw new Error('Invalid schedule time');
 return (+m[1]%12+(m[3]==='pm'?12:0))*60+ +(m[2]||0);
}
const out=[];
for (const item of $input.all()) {
 const row=item.json;
 if (!row.workflow || !row.timings || /^(Server|Python)::/i.test(row.workflow)) continue;
 if (parse(row.timings)!==minute) continue;
 const n=Number(row.row_number);
 if (!Number.isInteger(n) || n<2) throw new Error('Schedule row number missing');
 out.push({json:{row_number:n, workflow_name:row.workflow, scheduled_at:timestamp}});
}
return out;'''
RESOLVE=r'''const byName=new Map();
for (const item of $input.all()) {
 const w=item.json;
 if (!w.id || !w.name) throw new Error('Invalid n8n workflow inventory');
 if (!byName.has(w.name)) byName.set(w.name,new Map());
 byName.get(w.name).set(String(w.id),w);
}
return $('filter').all().map(item=>{
 const row=item.json, matches=byName.get(row.workflow_name);
 if (!matches || matches.size!==1) throw new Error('Schedule target missing or ambiguous: '+row.workflow_name);
 const id=matches.keys().next().value;
 if (id==='cSoy04ojAzkzBkVM') throw new Error('Scheduler cannot dispatch itself');
 return {json:{...row, workflow_id:id}};
});'''
UNWRAP_DISPATCH="return $input.all().filter(i=>i.json.allowed===true).map(i=>({json:i.json.payload}));"
UNWRAP_QUEUE="return $input.all().filter(i=>i.json.allowed===true).map(i=>({json:{...i.json.row,_sh01_guard:i.json.guard}}));"
HANDLER=r'''const q=$('Unwrap Queue Claim').first().json;
const raw=$input.first().json;
const http=Number(raw.statusCode ?? raw.status_code ?? raw.httpCode ?? 200);
let b=raw.body!==undefined?raw.body:raw;
if(typeof b==='string'){try{b=JSON.parse(b);}catch{b=null;}}
const fields=['status','claimed_at','render_job_id','render_token','render_video_url','render_b2_file_name','render_size_bytes','rendered_at','attempt_count','last_error'];
const out={queue_id:q.queue_id};
for(const key of fields) out[key]=q[key]??'';
const keep=(message)=>{out.last_error=message;return [{json:out}];};
if(http===409 || http===429 || http>=500) return keep('Transient renderer response; preserved job and retry count.');
if(http===404) return keep('Render job not found; retained for review, not resubmitted automatically.');
if(http!==200 && http!==202) return keep('Renderer request rejected; inspect access/configuration.');
if(!b || typeof b!=='object') return keep('Invalid renderer response; retained for review.');
const status=String(b.status||'').toLowerCase();
if(['queued','accepted','running'].includes(status)) {
 const id=b.job_id||q.render_job_id;
 if(!id) return keep('Pending response missing job identity; retained for review.');
 if(q.render_job_id && id!==q.render_job_id) return keep('Render job identity mismatch; retained for review.');
 out.status='rendering';out.render_job_id=id;
 out.claimed_at=q.claimed_at||new Date().toISOString();out.last_error='';
 return [{json:out}];
}
if(status==='complete' && b.video_url && b.token) {
 if(q.render_job_id && b.job_id && b.job_id!==q.render_job_id) return keep('Completed job identity mismatch; retained for review.');
 out.status='rendered';out.render_job_id=b.job_id||q.render_job_id||'';
 out.render_token=b.token;out.render_video_url=b.video_url;
 out.render_b2_file_name=b.b2_file_name||'';out.render_size_bytes=b.size_bytes??'';
 out.rendered_at=new Date().toISOString();out.last_error='';
 return [{json:out}];
}
if(['failed','interrupted'].includes(status)) {
 out.status='failed';out.attempt_count=Number(q.attempt_count||0)+1;
 out.last_error='Backend confirmed terminal render failure; manual review required.';
 return [{json:out}];
}
return keep('Unrecognized renderer response; job identity and retries preserved.');'''

def edge(name): return {'main': [[{'node': name, 'type':'main','index':0}]]}

def node(name, kind, parameters):
    return {'id':str(uuid.uuid5(uuid.NAMESPACE_URL,'sh01:'+REVISION+':'+name)),
            'name':name,'type':'n8n-nodes-base.'+kind,'typeVersion':2 if kind=='code' else 4.2,
            'position':[0,0],'parameters':parameters,'notes':REVISION}

def http(name,path,body,credential_node,base):
    n=node(name,'httpRequest',{'method':'POST','url':base+path,
        'authentication':'genericCredentialType','genericAuthType':'httpHeaderAuth',
        'sendBody':True,'specifyBody':'json','jsonBody':body,
        'options':{'timeout':60000,'response':{'response':{'responseFormat':'json'}}}})
    n['credentials']=copy.deepcopy(credential_node['credentials'])
    if not n['credentials'].get('httpHeaderAuth'): raise ValueError('Media header credential missing')
    n['retryOnFail']=False
    return n

def transform(workflows,base):
    out=copy.deepcopy(workflows)
    s,t=out[SCHEDULER],out[TELEGRAM]
    sn={n['name']:n for n in s['nodes']};tn={n['name']:n for n in t['nodes']}
    credential=tn['Submit Render Job']
    sn['filter']['parameters']['jsCode']=FILTER
    sn['n8n-workflows-data']['executeOnce']=True
    sn['n8n-workflows-data']['parameters']={'resource':'workflow','operation':'getAll','returnAll':True,'filters':{},'requestOptions':{}}
    sn['get-only-the-correct-workflow']['parameters']['jsCode']=RESOLVE
    sn['gen-payload']['parameters']['jsCode']=UNWRAP_DISPATCH
    sn['delegate-all-tasks']['parameters']['workflowId']={'__rl':True,'value':'={{ $json._sh01_workflow_id }}','mode':'id'}
    sn['delegate-all-tasks']['parameters']['mode']='each'
    sn['delegate-all-tasks']['parameters']['workflowInputs']={'mappingMode':'defineBelow','value':{},'schema':[]}
    sn['delegate-all-tasks']['retryOnFail']=False
    s['nodes'].append(http('Claim Scheduled Dispatch','/workflow-guards/schedule/claim',
                          '={{ JSON.stringify($json) }}',credential,base))
    s['connections']['get-only-the-correct-workflow']=edge('Claim Scheduled Dispatch')
    s['connections']['Claim Scheduled Dispatch']=edge('gen-payload')
    # Idle ticks must not contact the coordinator.
    tn['Select Queue Work']['parameters']['jsCode']=tn['Select Queue Work']['parameters']['jsCode'].replace("return [{json:{action:'idle'}}];",'return [];')
    for name in ['Poll Render Job','Submit Render Job']:
        params=tn[name]['parameters']
        for key,value in list(params.items()):
            if isinstance(value,str): params[key]=value.replace("$('Select Queue Work')","$('Unwrap Queue Claim')")
        tn[name]['retryOnFail']=False
    for name in ['Handle Poll Response','Handle Submit Response']:
        tn[name]['parameters']['jsCode']=HANDLER
    t['nodes'].append(http('Claim Queue Row','/workflow-guards/telegram/claim',
        '={{ JSON.stringify({queue_id: String($json.queue_id), action: $json.action}) }}',credential,base))
    t['nodes'].append(node('Unwrap Queue Claim','code',{'jsCode':UNWRAP_QUEUE}))
    replacement=http('Update Render Queue State','/workflow-guards/telegram/commit',
        "={{ JSON.stringify({guard: $('Unwrap Queue Claim').first().json._sh01_guard, queue_id: String($json.queue_id), changes: Object.fromEntries(Object.entries($json).filter(([k]) => k !== 'queue_id'))}) }}",credential,base)
    t['nodes']=[replacement if n['name']=='Update Render Queue State' else n for n in t['nodes']]
    t['connections']['Select Queue Work']=edge('Claim Queue Row')
    t['connections']['Claim Queue Row']=edge('Unwrap Queue Claim')
    t['connections']['Unwrap Queue Claim']=edge('Route Queue Action')
    # Disable automatic retries on every side-effecting added node.
    for w in out.values():
        for n in w['nodes']:
            if n['name'] in ('Claim Scheduled Dispatch','Claim Queue Row','Update Render Queue State'):
                n.pop('onError',None);n['continueOnFail']=False
    return out
