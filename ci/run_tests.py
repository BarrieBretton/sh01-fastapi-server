"""Run every offline unittest; absent FFmpeg or skipped regressions must fail CI."""
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for executable in ('ffmpeg', 'ffprobe'):
    if not shutil.which(executable):
        raise SystemExit(f'Missing required executable: {executable}')
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.testsRun or result.skipped or not result.wasSuccessful():
    raise SystemExit('CI requires a nonempty passing suite with no skipped tests.')
