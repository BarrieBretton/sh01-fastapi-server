"""Compare shipped Python/notebook and embedded modules without running the cell.

Normalize CRLF so Windows builds compare correctly on Linux runners. Also verify
bootstrap/template parity; a stale template is as important as stale modules.
"""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'notebooks/sh01_full_editor_one_cell.py').read_text()
notebook = json.loads((ROOT / 'notebooks/sh01_full_editor_one_cell.ipynb').read_text())
assert source == ''.join(c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
                         for c in notebook['cells'] if c['cell_type'] == 'code'), 'Notebook/Python differ'
import nbformat
nbformat.validate(nbformat.from_dict(notebook))
tree = ast.parse(source)
assignments = {n.targets[0].id: n for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name)}
bundle = ast.literal_eval(assignments['BUNDLE'].value)
digest = ast.literal_eval(assignments['BUNDLE_SHA256'].value)
raw = base64.b85decode(bundle)
assert hashlib.sha256(raw).hexdigest() == digest, 'Bundle hash mismatch'
sys.path.insert(0, str(ROOT / 'scripts'))
from build_one_cell import FILES
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert sorted(archive.namelist()) == sorted(FILES), 'Bundled module inventory differs'
    for name in FILES:
        embedded = archive.read(name).decode().replace('\r\n', '\n')
        assert embedded == (ROOT / name).read_text(), f'Stale embedded module: {name}'
# Replace generated assignments by template placeholders, then compare setup code.
lines = source.splitlines(keepends=True)
for name, replacement in sorted([('BUNDLE', "BUNDLE = '__SOURCE_BUNDLE__'\n"),
                                 ('BUNDLE_SHA256', "BUNDLE_SHA256 = '__SOURCE_SHA256__'\n")],
                                key=lambda pair: assignments[pair[0]].lineno, reverse=True):
    node = assignments[name]
    lines[node.lineno - 1:node.end_lineno] = [replacement]
expected = (ROOT / 'scripts/kaggle_one_cell_template.py').read_text().replace(
    '# __ENVIRONMENT_BOOTSTRAP__', (ROOT / 'scripts/kaggle_bootstrap.py').read_text())
assert ''.join(lines) == expected, 'Stale bootstrap/template; run python scripts/build_one_cell.py'
print('Notebook schema, template, hash, and all embedded modules match.')
