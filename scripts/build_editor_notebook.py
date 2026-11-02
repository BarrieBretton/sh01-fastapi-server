"""Generate the compact notebook from readable, syntax-checkable cell sources."""
from pathlib import Path
import ast
import json
import textwrap

cells = []


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    if kind == 'code' and not source.startswith('%%'):
        ast.parse(source)
    value = dict(cell_type=kind, id=f'editor-{len(cells):02d}', metadata={}, source=source.splitlines(True))
    if kind == 'code':
        value.update(outputs=[], execution_count=None)
    cells.append(value)


cell('markdown', '''
# SH01 automatic video editor
Run top to bottom with **GPU and Internet enabled**. The main run accepts a source
and a short brief; it plans narrative and supporting visuals, renders drafts, checks
them, and keeps a valid result. This uses the existing `/media/ai-compose` endpoint.

This notebook does not test every external provider. FFmpeg renders animated cards,
checklists/timelines, kinetic captions, camera tracks, overlays and audio. It does not
run Remotion or generate arbitrary 3D scenes. The visual reviewer downloads
Qwen2.5-VL-3B locally; text and vision models run sequentially to reduce GPU memory
use. Piper narration needs a separately installed voice model.

Your private `.env` stays in the notebook. Existing `.env` files are preserved.
''')
cell('code', '''
import os, subprocess, sys
from pathlib import Path

REPO = Path('/kaggle/working/sh01-fastapi-server')
REF = os.environ.get('AI_COMPOSE_REPO_REF', 'feature/ai-compose-kaggle')
URL = os.environ.get('AI_COMPOSE_REPO_URL', 'https://github.com/BarrieBretton/sh01-fastapi-server.git')
def run(*args, **kwargs):
    """Run a command and stop setup immediately if it fails."""
    return subprocess.run(list(args), check=True, **kwargs)

if (REPO / '.git').exists():
    dirty = subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'], text=True).strip()
    if dirty:
        raise RuntimeError('Checkout has local edits. Preserve/commit them before updating; nothing was overwritten.\\n'+dirty)
    run('git','-C',str(REPO),'fetch','origin',REF)
    run('git','-C',str(REPO),'switch',REF)
    run('git','-C',str(REPO),'merge','--ff-only',f'origin/{REF}')
else:
    run('git','clone','--branch',REF,URL,str(REPO))
run('git','-C',str(REPO),'log','-1','--oneline')
assert (REPO/'ai_compose_editor.py').is_file(), 'This checkout lacks the automatic editor'
''')
cell('markdown', '''
## Install and check the GPU
Preserve Kaggle's installed Torch/CUDA combination. The fresh subprocess check also
catches a broken installation without importing Torch into the notebook kernel.
If this fails, restart the Kaggle session with GPU enabled before continuing.
''')
cell('code', '''
import importlib.metadata
run('nvidia-smi')
constraints = Path('/kaggle/working/editor-constraints.txt')
constraints.write_text('torch=='+importlib.metadata.version('torch')+'\\n')
run(sys.executable,'-m','pip','install','-q','-c',str(constraints),
    'fastapi>=0.115','uvicorn','requests','httpx','python-dotenv','pydantic>=2.9',
    'Pillow','numpy','faster-whisper','-r',str(REPO/'requirements-ai-compose.txt'))
run(sys.executable,'-c',
    'import torch; print(torch.__version__, torch.version.cuda); '
    'assert torch.cuda.is_available(), "CUDA unavailable"; '
    'print(torch.cuda.get_device_name(0)); import bitsandbytes')
run('ffmpeg','-version', stdout=subprocess.DEVNULL)
run(sys.executable,'-m','unittest','discover','-s','tests','-p','test_ai_compose*.py','-v',
    cwd=REPO, env={**os.environ,'MEDIA_ALLOW_UNLIMITED_RESOURCES':'true',
                  'MEDIA_FFMPEG_MAX_RSS_MB':'2048','MEDIA_MIN_START_HEADROOM_MB':'64'})
''')
cell('markdown', '''
## Private environment values
Edit the heredoc before the first run, or edit the existing `.env`. Empty provider
keys mean those services are unavailable. For B-roll, mount an annotated asset library
or provide Pexels/Pixabay keys. Source and assets are different directories.
''')
cell('code', r'''
%%bash
set -e
cd /kaggle/working/sh01-fastapi-server
if [ ! -f .env ]; then
cat > .env <<'ENV'
AI_COMPOSE_SOURCE_PATH=/kaggle/input/datasets/vivanjaiswal/video1/military_seth_raw_cut1.mp4
MEDIA_CLIP_API_KEY=
AI_COMPOSE_WORK_DIR=/kaggle/working/ai-compose-jobs
AI_COMPOSE_HF_MODEL=Qwen/Qwen2.5-3B-Instruct
AI_COMPOSE_HF_MAX_NEW_TOKENS=2600
AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY=false
MEDIA_CLIP_WHISPER_MODEL=small
AI_COMPOSE_ASSET_LIBRARY=
PEXELS_API_KEY=
PIXABAY_API_KEY=
UNSPLASH_ACCESS_KEY=
AI_COMPOSE_MUSIC_PATH=
AI_COMPOSE_TTS_MODEL=
AI_COMPOSE_VISION_HF_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
AI_COMPOSE_VISION_URL=
AI_COMPOSE_VISION_MODEL=
ENV
else
  echo 'Existing .env retained. Edit that file if settings need to change.'
fi
''')
cell('code', '''
from dotenv import load_dotenv
import json, secrets, time, requests, numpy as np
from IPython.display import Audio, Video, FileLink, display

load_dotenv(REPO/'.env', override=True)
os.environ.setdefault('AI_COMPOSE_VISION_HF_MODEL','Qwen/Qwen2.5-VL-3B-Instruct')
if not os.environ.get('MEDIA_CLIP_API_KEY','').strip():
    os.environ['MEDIA_CLIP_API_KEY'] = secrets.token_urlsafe(32)
source_setting = os.environ.get('AI_COMPOSE_SOURCE_PATH','').strip()
if not source_setting:
    raise ValueError('Set AI_COMPOSE_SOURCE_PATH in .env (no trailing space in its name)')
SOURCE = Path(source_setting).resolve()
assert SOURCE.is_file(), f'Mount the input video first: {SOURCE}'
os.environ['MEDIA_LOCAL_SOURCE_DIR'] = str(SOURCE.parent)
os.environ['MEDIA_ALLOW_UNLIMITED_RESOURCES'] = 'true'
os.environ['PUBLIC_API_BASE'] = 'http://127.0.0.1:5000'
BASE = os.environ['PUBLIC_API_BASE']
HEADERS = {'X-API-KEY': os.environ['MEDIA_CLIP_API_KEY']}

def beep():
    """Play a short alert; browser autoplay settings may require interaction."""
    sample_rate = 8000
    t = np.linspace(0, .5, int(sample_rate*.5), endpoint=False)
    display(Audio(np.sin(2*np.pi*440*t), rate=sample_rate, autoplay=True))

print('Source:', SOURCE)
beep()
''')
cell('markdown', '''
## Start the editor
One background worker, with environment values copied **before** startup. Changing
`.env` requires rerunning the environment cell and this cell. SQLite stores notebook
job receipts; restarting interrupts unfinished jobs. Production services are not started.
''')
cell('code', '''
if 'server' in globals() and server.poll() is None:
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill(); server.wait()
import socket
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1',5000)) == 0:
        raise RuntimeError('Port 5000 belongs to another process; stop that server before proceeding')
LOG = Path('/kaggle/working/sh01_editor.log')
with LOG.open('w') as log:
    server = subprocess.Popen([sys.executable,'-m','uvicorn','ai_compose_local_app:app',
        '--host','127.0.0.1','--port','5000','--workers','1'], cwd=REPO,
        env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
for _ in range(90):
    if server.poll() is not None:
        raise RuntimeError('Server exited. Inspect '+str(LOG))
    try:
        response = requests.get(BASE+'/media/ai-compose-capabilities',headers=HEADERS,timeout=3)
        response.raise_for_status()
        caps = response.json()
        assert 'automatic-editor-v1' in caps['features'], 'Stale server process'
        break
    except requests.ConnectionError:
        time.sleep(1)
else:
    raise TimeoutError('Server did not become ready; inspect '+str(LOG))
print('Server ready; PID:',server.pid)
print('Local AI configured:',caps['local_director'])
print('Provider availability:',caps['asset_sources'])
''')
cell('markdown', '''
## Edit your video — one integrated run
Change the brief and target duration. The editor decides visual/audio events within
your allowed capabilities. Existing dialogue is preserved; no added narration over it.
Music `auto` uses a configured track, otherwise stays off. Set `procedural` explicitly
for a simple original synthesized bed. This is not an AI music generator.

When a B-roll source is available, the default requires at least one resolved layer:
a plain captioned clip cannot silently pass that test. Set `REQUIRE_BROLL=False` for
an intentional graphics-only edit. No configured provider means no remote retrieval.
''')
cell('code', '''
BRIEF = ('Create a polished informative short. Choose a coherent opening and payoff; '
         'preserve factual meaning and original speech. Add concise animated visual explanations, '
         'relevant supporting footage where useful, readable highlighted captions, and restrained sound design.')
TARGET_SECONDS = 30
REQUIRE_BROLL = bool(os.environ.get('AI_COMPOSE_ASSET_LIBRARY') or
                     any(os.environ.get(k) for k in ('PEXELS_API_KEY','PIXABAY_API_KEY','UNSPLASH_ACCESS_KEY')))
import uuid
payload = {
    'request_id':'editor-'+str(uuid.uuid4()), 'source_path':str(SOURCE),
    'brief':BRIEF, 'unlimited_resources':True, 'target_seconds':TARGET_SECONDS,
    'preset':'viral_short', 'output_count':1, 'llm_required':True,
    'captions_enabled':True, 'fetch_assets':True, 'refinement_iterations':1,
    'local_asset_library':os.environ.get('AI_COMPOSE_ASSET_LIBRARY') or None,
    'asset_sources':{'local_library':True,'pexels':True,'pixabay':True,
                     'unsplash':bool(os.environ.get('UNSPLASH_ACCESS_KEY'))},
    'visual_model':os.environ.get('AI_COMPOSE_VISION_MODEL') or None,
    'editor':{'enabled':True,'caption_style':'highlight','min_graphics':2,
              'min_visual_layers':1 if REQUIRE_BROLL else 0,'music':'auto',
              'strict_requirements':True,'visual_review_required':True,'publish_outputs':False},
    'editorial_policy':{'supplementary_visuals':'require' if REQUIRE_BROLL else 'auto',
                        'camera_motion':'dynamic','motion_graphics':'selective'},
}
if not REQUIRE_BROLL:
    print('No B-roll source configured: testing graphics/captions/camera/audio. Add a library or key for B-roll.')
response = requests.post(BASE+'/media/ai-compose',headers=HEADERS,json=payload,timeout=60)
if not response.ok:
    raise RuntimeError(f'Submit {response.status_code}: {response.text[:2000]}')
job = response.json()
print('Job:',job['job_id'])
deadline = time.monotonic()+7200
previous = None
while time.monotonic() < deadline:
    response = requests.get(BASE+'/media/ai-compose-jobs/'+job['job_id'],headers=HEADERS,timeout=60)
    response.raise_for_status()
    status = response.json()
    state = (status.get('status'),status.get('stage'),status.get('refinement_round'),status.get('progress'))
    if state != previous:
        print(state); previous = state
    if status.get('status') in {'complete','failed','interrupted'}:
        break
    time.sleep(5)
else:
    raise TimeoutError('Polling timed out; job may still be running. Resume polling this job_id, do not resubmit.')
beep()
if status.get('status') != 'complete':
    raise RuntimeError(json.dumps(status,indent=2))
''')
cell('markdown', '''
## Watch, inspect decisions, and save results
Compare the actual video with the brief. Counts verify execution, not artistic quality
or viral potential. The receipt lists unavailable capabilities and review results.
The ZIP excludes `.env`, API logs and input videos. Kaggle cannot silently save files
to your laptop or bypass the browser's download confirmation; save notebook outputs
or download this ZIP before the session disappears.
''')
cell('code', '''
import zipfile
receipt = status['plan']
print('Winning draft:',receipt['winning_round'])
print('Warnings:',json.dumps(receipt['warnings'],indent=2))
print('Visual review:',receipt['quality_review'].get('visual'))
print('Graphics:',[g['kind'] for g in receipt['plan']['graphics']])
print('Supporting layers:',len(receipt['resolved_layers']))
for output in status['outputs']:
    print('Executed:',output.get('render_features'))
    display(Video(output['local_path'],embed=True,width=360))
root = Path(status['work_dir'])
archive = Path('/kaggle/working') / (job['job_id']+'-editor-results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and (path.name.startswith('variant_') or
                              path.name in {'composition_receipt.json','evidence.json','review_history.json','storyboard.json'}):
            bundle.write(path,path.relative_to(root))
display(FileLink(str(archive)))
beep()
''')
cell('markdown', '''
## Stop when finished
Run after the ZIP is created. Reuse the same running server to submit another prompt
by rerunning the integrated edit cell. No need to reinstall packages for each video.
''')
cell('code', '''
server.terminate()
try:
    server.wait(timeout=15)
except subprocess.TimeoutExpired:
    server.kill(); server.wait()
print('Server stopped. Results:',archive)
''')

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]/'notebooks/sh01_automatic_editor.ipynb'
    path.parent.mkdir(exist_ok=True)
    notebook = dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.13.0'}}, nbformat=4, nbformat_minor=5)
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(path, 'cells:',len(cells))
