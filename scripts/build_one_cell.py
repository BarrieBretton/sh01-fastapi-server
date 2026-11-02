"""Package reviewed repository modules into one reproducible Kaggle code cell.

The embedded archive contains Python source, not opaque bytecode. Users can inspect
it at /kaggle/working/sh01-onecell-*/code after running the cell.
"""
from pathlib import Path
import ast
import base64
import hashlib
import io
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=sorted(p.name for p in ROOT.glob('ai_compose*.py')) + ['media_transcribe.py','media_resources.py','media_job_store.py']

def build():
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in FILES:
            data=(ROOT/name).read_bytes()
            ast.parse(data,filename=name)
            item=zipfile.ZipInfo(name,date_time=(2026,11,2,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=0o100644 << 16
            archive.writestr(item,data)
    raw=payload.getvalue();digest=hashlib.sha256(raw).hexdigest()
    encoded=base64.b85encode(raw).decode()
    # Consecutive string literals keep the single cell readable in notebook editors.
    literal='(\n'+ '\n'.join(repr(encoded[i:i+110]) for i in range(0,len(encoded),110))+'\n)'
    source=(ROOT/'scripts/kaggle_one_cell_template.py').read_text()
    bootstrap=(ROOT/'scripts/kaggle_bootstrap.py').read_text()
    ast.parse(bootstrap)
    source=source.replace('# __ENVIRONMENT_BOOTSTRAP__', bootstrap)
    source=source.replace("'__SOURCE_BUNDLE__'",literal).replace('__SOURCE_SHA256__',digest)
    ast.parse(source)
    target=ROOT/'notebooks/sh01_full_editor_one_cell.py';target.write_text(source)
    notebook={'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python'},'sh01':{'base_commit':'fed2f61f15d91bdb8feaa94e2c17550a98160023','bundle_sha256':digest}},
        'cells':[{'cell_type':'code','id':'sh01-full-editor','execution_count':None,'metadata':{},'outputs':[],
                  'source':source.splitlines(True)}]}
    (ROOT/'notebooks/sh01_full_editor_one_cell.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    print(f'Built one cell: {len(FILES)} Python modules, {len(raw):,} compressed bytes, SHA256 {digest}')
    return digest

if __name__=='__main__': build()
