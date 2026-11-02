"""One-job notebook driver using the existing local FastAPI application.

Runs in a fresh process, reads private configuration from a file, and never prints
keys. The server receives its complete environment before importing any modules.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile


def request_json(base, path, key, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base+path, data=data, headers={'X-API-KEY':key,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode(errors='replace')[:1500]
        raise RuntimeError(f'HTTP {exc.code} from {path}: {detail}') from exc


def archive_run(root, outputs):
    """Export finals and review/provenance data, excluding environment values."""
    credits=[]
    receipt=root/'composition_receipt.json'
    if receipt.is_file():
        for asset in json.loads(receipt.read_text()).get('asset_manifest',[]):
            credits.append('\n'.join(str(asset.get(k) or '') for k in
                ('asset_id','title','attribution','license','license_url','source_url')))
    (root/'ASSET_CREDITS.txt').write_text('\n\n'.join(credits) or 'No external assets used.\n')
    archive=root.parent/(root.name+'-output.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for name in ('composition_receipt.json','evidence.json','review_history.json','ASSET_CREDITS.txt','status.json','request.json','runtime.json'):
            path=root/name
            if path.is_file():
                output.write(path,name)
        for item in outputs:
            path=Path(item['local_path'])
            if path.is_file() and path.parent.resolve()==root.resolve():
                output.write(path,path.name)
    return archive


def main(config_path):
    config=json.loads(Path(config_path).read_text())
    work=Path(config['work_dir']); work.mkdir(parents=True,exist_ok=True)
    source=Path(config['source_path']).expanduser().resolve(strict=True)
    environment=os.environ.copy()
    env_file=Path(config.get('env_file') or '/nonexistent')
    if env_file.is_file():
        from dotenv import dotenv_values
        environment.update({k:v for k,v in dotenv_values(env_file).items() if v is not None})
    environment.update({str(k):str(v) for k,v in config.get('environment',{}).items()})
    key=environment.get('MEDIA_CLIP_API_KEY','').strip() or secrets.token_urlsafe(32)
    environment.update(MEDIA_CLIP_API_KEY=key,MEDIA_LOCAL_SOURCE_DIR=str(source.parent),
        MEDIA_ALLOW_UNLIMITED_RESOURCES='true',AI_COMPOSE_WORK_DIR=str(work),
        AI_COMPOSE_LOCAL_FALLBACK='true',AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY='false')
    # Bind only to loopback; no public tunnel, production cron, or cloud queue.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    log_path=work/'server.log'
    log=log_path.open('a')
    process=subprocess.Popen([sys.executable,'-m','uvicorn','ai_compose_local_app:app',
        '--host','127.0.0.1','--port',str(port),'--workers','1'],
        cwd=Path(__file__).parent,env=environment,stdout=log,stderr=subprocess.STDOUT)
    outputs=[]; job_root=None; successful=False
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('Local API failed to start; inspect '+str(log_path))
            try:
                capabilities=request_json(base,'/media/ai-compose-capabilities',key)
                break
            except (OSError,RuntimeError):
                time.sleep(1)
        else:
            raise TimeoutError('Local API startup timed out; inspect '+str(log_path))
        if 'measured-narration-sections' not in capabilities.get('features',[]):
            raise RuntimeError('Server loaded stale code: narration sections are unavailable')
        payload=dict(config['payload'],source_path=str(source),unlimited_resources=True,
                     request_id='onecell-'+secrets.token_hex(12))
        job=request_json(base,'/media/ai-compose',key,payload)
        job_id=job['job_id'];job_root=work/job_id;job_root.mkdir(exist_ok=True)
        # No API credentials are part of the payload or artifact archive.
        (job_root/'request.json').write_text(json.dumps(payload,indent=2))
        from importlib.metadata import version, PackageNotFoundError
        versions={}
        for name in ('torch','transformers','bitsandbytes','faster-whisper','pydantic','fastapi'):
            try: versions[name]=version(name)
            except PackageNotFoundError: versions[name]='missing'
        (job_root/'runtime.json').write_text(json.dumps({'python':sys.version,'packages':versions,
            'models':{k:environment.get(k) for k in ('AI_COMPOSE_HF_MODEL','AI_COMPOSE_VISION_HF_MODEL','AI_COMPOSE_TTS_MODEL')}},indent=2))
        print('Job:',job_id,'| local API:',base,flush=True)
        deadline=time.monotonic()+int(config.get('timeout_seconds',7200))
        previous=None; last_report=0
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('Local API stopped; inspect '+str(log_path))
            status=request_json(base,'/media/ai-compose-jobs/'+job_id,key)
            temporary=job_root/'status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(job_root/'status.json')
            label=(status.get('stage'),status.get('refinement_round'),status.get('progress'))
            if label!=previous or time.monotonic()-last_report>=45:
                print('stage=%s round=%s progress=%s'%label,flush=True)
                previous=label;last_report=time.monotonic()
            if status.get('status')=='failed':
                raise RuntimeError(status.get('error') or 'Editor failed')
            if status.get('status')=='complete':
                outputs=status.get('outputs',[])
                if not outputs:
                    raise RuntimeError('Complete job returned no files')
                successful=True;break
            time.sleep(3)
        else:
            raise TimeoutError('Editor exceeded configured timeout; draft evidence retained in '+str(job_root))
        archive=archive_run(job_root,outputs)
        summary={'job_id':job_id,'outputs':outputs,'archive':str(archive),'receipt':str(job_root/'composition_receipt.json')}
        if config.get('keep_api'):
            connection=work/'local-api-private.json'
            connection.write_text(json.dumps({'base_url':base,'headers':{'X-API-KEY':key},'pid':process.pid}))
            connection.chmod(0o600)
            summary['api_connection_file']=str(connection)
        (work/'latest-output.json').write_text(json.dumps(summary,indent=2))
        print('Complete. Files:',', '.join(item['local_path'] for item in outputs),flush=True)
        print('Feature counts:',json.dumps(outputs[0].get('render_features',{})),flush=True)
        print('Archive:',archive,flush=True)
    finally:
        try:
            if job_root is not None and not successful:
                archive_run(job_root,outputs)
        finally:
            if not (successful and config.get('keep_api')):
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=10)
            log.close()


if __name__=='__main__':
    try:
        main(sys.argv[1])
    except Exception as exc:
        print(f'Editor stopped: {type(exc).__name__}: {exc}',flush=True)
        raise
