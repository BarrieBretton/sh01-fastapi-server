"""Install the template's actual Whisper/PyAV constraints; decode without a model."""
import ast
import io
from pathlib import Path
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'scripts/kaggle_one_cell_template.py').read_text())
requirements = sorted({node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and node.value.startswith(('faster-whisper>', 'av=', 'av<', 'av>'))})
if not any(r.startswith('faster-whisper') for r in requirements):
    raise SystemExit('Whisper requirement missing from notebook template')
subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
from faster_whisper.audio import decode_audio
buffer = io.BytesIO()
with wave.open(buffer, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b'\x00\x00' * 16000)
buffer.seek(0)
assert len(decode_audio(buffer, sampling_rate=16000)) == 16000
print('Actual faster-whisper audio decoder passed; no model downloaded.')
