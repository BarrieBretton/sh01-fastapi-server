# SH01 integrated editor — paste this entire file into ONE Kaggle Python cell.
# Enable GPU + Internet in Kaggle. Models download on the first run; no paid API key.
SOURCE_PATH = '/kaggle/input/datasets/vivanjaiswal/video1/military_seth_raw_cut1.mp4'
BRIEF = ('Create a polished, informative vertical short. Select a coherent story and a strong opening; '
         'preserve the speaker’s meaning and voice. Add brief grounded narration in separate intro, bridge '
         'and outro sections. Use relevant supplementary images or video, readable motion graphics, '
         'purposeful camera moves and transitions, restrained SFX and a quiet music bed.')
TARGET_SECONDS = 35
NARRATION = 'intro_bridge_outro'   # 'intro_outro', 'intro', 'auto', or 'off'
REQUIRED_FEATURES = ['voiceover','sfx','music','camera_motion','transitions','graphics','supplementary_media','captions']
REFINEMENT_ROUNDS = 2
OUTPUT_SIZE = (1080, 1920)       # Use (720,1280) for a faster draft.
LOCAL_ASSET_LIBRARY = ''        # Optional annotated media directory; uses Commons otherwise.
MUSIC_PATH = ''                 # Optional local track; empty => original procedural ambient bed.
PRIVATE_ENV_FILE = '/kaggle/working/sh01-fastapi-server/.env'  # Read if present; never overwrite.
TEXT_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
VISION_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
VOICE_NAME = 'en_US-ljspeech-high'  # English public voice; change for another language.
ASR_DEVICE = 'cpu'             # 'cuda' is optional; director and vision use the GPU either way.
KEEP_LOCAL_API = False         # True leaves the loopback API running after this job.
TIMEOUT_SECONDS = 7200

# -------------------- automated setup and execution --------------------
import base64, hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, urllib.request, venv, zipfile
from pathlib import Path

# Embedded, inspectable Python source from the matching repository patch.
BUNDLE = '__SOURCE_BUNDLE__'
BUNDLE_SHA256 = '__SOURCE_SHA256__'
raw = base64.b85decode(BUNDLE)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256, 'Damaged code bundle'
RUNTIME = Path('/kaggle/working') / ('sh01-onecell-' + BUNDLE_SHA256[:12])
CODE = RUNTIME / 'code'
CODE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        target = CODE / name
        if not target.resolve().is_relative_to(CODE.resolve()) or not name.endswith('.py'):
            raise RuntimeError('Unexpected file in source bundle')
        data = archive.read(name)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Bundled code was locally edited: {target}; preserve it before rerunning')
        target.write_bytes(data)
source = Path(SOURCE_PATH).expanduser()
if not source.is_file():
    raise FileNotFoundError(f'Mount the video and set SOURCE_PATH to the exact filename: {source}')
if shutil.which('nvidia-smi') is None:
    raise RuntimeError('Enable a Kaggle GPU accelerator before running this cell')
subprocess.run(['nvidia-smi'], check=True)

PYTHON = RUNTIME / 'venv' / 'bin' / 'python'
if not PYTHON.exists():
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(PYTHON.parent.parent)
# Use a private environment; preserve Kaggle's CUDA-enabled Torch installation.
import importlib.metadata
constraints = RUNTIME / 'constraints.txt'
constraints.write_text('torch==' + importlib.metadata.version('torch') + '\n')
setup_log = RUNTIME / 'setup.log'

def run_setup(command, *, env=None):
    """Record setup failures without putting secrets or huge pip logs in output."""
    with setup_log.open('a') as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        print(setup_log.read_text(errors='replace')[-4500:])
        raise RuntimeError(f'Setup failed; full log: {setup_log}')

stamp = RUNTIME / 'dependencies-ready'
if not stamp.exists():
    print('Installing isolated editor dependencies…', flush=True)
    run_setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-q','-c',constraints,
        'fastapi>=0.115,<1','uvicorn>=0.30,<1','pydantic>=2.9,<3','requests>=2.32,<3',
        'httpx>=0.27,<1','python-dotenv>=1,<2','Pillow>=10','numpy>=1.26',
        'transformers>=4.51,<5','accelerate>=1,<2','bitsandbytes>=0.45,<1',
        'json-repair>=0.30,<1','faster-whisper>=1.1,<2','sentencepiece'])
    stamp.write_text('Installed; Torch kept at '+importlib.metadata.version('torch'))
check = subprocess.check_output([str(PYTHON),'-c',
    'import torch, bitsandbytes, transformers, json, pathlib, site; '
    'assert torch.cuda.is_available(), "CUDA unavailable in fresh subprocess"; '
    'roots=[pathlib.Path(p) for p in site.getsitepackages()]; '
    'libs=[str(p) for root in roots for p in root.glob("nvidia/*/lib")]; '
    'print(json.dumps({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"libraries":libs}))'],text=True)
gpu = json.loads(check.strip().splitlines()[-1])
print('GPU ready:', gpu['gpu'], '| Torch', gpu['torch'], flush=True)
if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
    run_setup(['apt-get','update','-qq'])
    run_setup(['apt-get','install','-y','-qq','ffmpeg','fonts-dejavu-core'])

MODELS = Path('/kaggle/working/sh01-editor-models'); MODELS.mkdir(exist_ok=True)
def download(url, path):
    """Atomic public model/binary download; retry transient network errors."""
    if path.is_file() and path.stat().st_size:
        return
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'SH01LocalEditor/2.0'})
            with urllib.request.urlopen(request,timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response,out,1024*1024)
            partial.replace(path);return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt==2: raise
            time.sleep(2**attempt)

# The standalone Piper distribution avoids Python-version wheel mismatches.
PIPER = MODELS / 'piper' / 'piper'
if NARRATION != 'off' or 'voiceover' in REQUIRED_FEATURES:
    if not PIPER.is_file():
        print('Installing local narration engine…',flush=True)
        packed=MODELS/'piper_amd64.tar.gz'
        download('https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz',packed)
        with tarfile.open(packed) as archive:
            archive.extractall(MODELS,filter='data')
        PIPER.chmod(0o755)
    locale, speaker, quality = VOICE_NAME.split('-')
    language = locale.split('_')[0]
    voice_url=f'https://huggingface.co/rhasspy/piper-voices/resolve/main/{language}/{locale}/{speaker}/{quality}/'
    voice=MODELS/(VOICE_NAME+'.onnx')
    for name in (VOICE_NAME+'.onnx',VOICE_NAME+'.onnx.json'):
        download(voice_url+name,MODELS/name)
    download(voice_url+'MODEL_CARD',MODELS/(VOICE_NAME+'-MODEL_CARD.txt'))
    run_setup([PIPER,'--help'])
else:
    voice=MODELS/(VOICE_NAME+'.onnx')

WORK = Path('/kaggle/working/ai-compose-onecell-jobs') / time.strftime('%Y%m%d-%H%M%S')
WORK.mkdir(parents=True,exist_ok=True)
environment = {
    'PATH':str(PIPER.parent)+os.pathsep+os.environ.get('PATH',''),
    'HF_HOME':str(MODELS/'huggingface'),
    'AI_COMPOSE_HF_MODEL':TEXT_MODEL,'AI_COMPOSE_HF_LOAD_4BIT':'true',
    'AI_COMPOSE_HF_MAX_NEW_TOKENS':'3000','AI_COMPOSE_VISION_HF_MODEL':VISION_MODEL,
    'MEDIA_CLIP_WHISPER_MODEL':'small','MEDIA_CLIP_WHISPER_DEVICE':ASR_DEVICE,
    'MEDIA_CLIP_WHISPER_COMPUTE_TYPE':'float16' if ASR_DEVICE=='cuda' else 'int8',
    'MEDIA_CLIP_WHISPER_CPU_THREADS':'4','MEDIA_CLIP_WHISPER_CACHE_DIR':str(MODELS/'whisper'),
    'AI_COMPOSE_TTS_MODEL':str(voice),'AI_COMPOSE_MUSIC_PATH':MUSIC_PATH,
    'LD_LIBRARY_PATH':os.pathsep.join(gpu['libraries']+[os.environ.get('LD_LIBRARY_PATH','')]),
    'AI_COMPOSE_USE_NVENC':'false','AI_COMPOSE_VIDEO_CODEC':'libx264',
    'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
}
payload = {
    'brief':BRIEF,'preset':'viral_short','target_seconds':TARGET_SECONDS,
    'output_width':OUTPUT_SIZE[0],'output_height':OUTPUT_SIZE[1], 'output_count':1,
    'captions_enabled':True,'fetch_assets':True,'llm_required':True,
    'refinement_iterations':REFINEMENT_ROUNDS,'local_asset_library':LOCAL_ASSET_LIBRARY or None,
    'asset_sources':{'local_library':True,'wikimedia':True,'pexels':True,'pixabay':True},
    'editorial_policy':{'source_audio':'preserve','supplementary_visuals':'require',
        'camera_motion':'dynamic','motion_graphics':'hero'},
    'editor':{'enabled':True,'narration':NARRATION,'narration_budget_seconds':12,
        'caption_style':'highlight','theme':'editorial','progress_bar':True,
        'music':'library' if MUSIC_PATH else 'procedural','procedural_sfx':True,
        'transition':'dissolve','min_graphics':2,'min_visual_layers':1,
        'required_features':REQUIRED_FEATURES,'visual_review_required':True,
        'strict_requirements':True,'publish_outputs':False},
    'sfx':{'mode':'adaptive','volume':0.20}, 'music_volume':0.12,
}
config={'source_path':str(source),'env_file':PRIVATE_ENV_FILE,'environment':environment,
    'work_dir':str(WORK),'payload':payload,'timeout_seconds':TIMEOUT_SECONDS,'keep_api':KEEP_LOCAL_API}
config_path=WORK/'private-config.json';config_path.write_text(json.dumps(config,indent=2));config_path.chmod(0o600)
# Run outside the notebook kernel: no stale imports and no foreground Uvicorn cell.
from IPython.display import Audio, FileLink, Video, display
import numpy as np

def beep():
    """Notify at readiness, completion, and failure; browser autoplay rules apply."""
    t=np.linspace(0,.5,4000,endpoint=False)
    display(Audio(np.sin(2*np.pi*440*t),rate=8000,autoplay=True))

beep()
print('Running the integrated editor. Evidence, drafts, and logs are saved under:',WORK,flush=True)
runner_log=WORK/'driver.log'
with runner_log.open('w') as stream:
    process=subprocess.Popen([str(PYTHON),'-u',str(CODE/'ai_compose_kaggle.py'),str(config_path)],
        cwd=CODE,stdout=stream,stderr=subprocess.STDOUT)
    offset=0
    try:
        while process.poll() is None:
            time.sleep(2)
            with runner_log.open() as current:
                current.seek(offset);message=current.read();offset=current.tell()
            if message: print(message,end='',flush=True)
    except KeyboardInterrupt:
        process.send_signal(2)
        try: process.wait(timeout=25)
        except subprocess.TimeoutExpired: process.kill();process.wait()
        beep()
        raise
with runner_log.open() as current:
    current.seek(offset);print(current.read(),end='')
beep()
if process.returncode:
    display(FileLink(str(runner_log)))
    display(FileLink(str(WORK/'server.log')))
    for diagnostic in WORK.glob('*-output.zip'): display(FileLink(str(diagnostic)))
    raise RuntimeError('No passing final render was produced. Review the saved diagnostics; requirements were not silently disabled.')
result=json.loads((WORK/'latest-output.json').read_text())
for output in result['outputs']:
    display(Video(output['local_path'],embed=True,width=360))
    display(FileLink(output['local_path']))
display(FileLink(result['archive']))
display(FileLink(result['receipt']))
if 'api_connection_file' in result:
    print('Loopback API remains available. Connection details:',result['api_connection_file'])
print('Save a Kaggle notebook version with outputs to retain these files after the session ends.')
