"""Exercise the builder under both UTF-8 and Windows legacy text defaults."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class NotebookEncodingTests(unittest.TestCase):
    def test_builder_preserves_unicode_with_platform_defaults(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/build_one_cell.py'
        spec = importlib.util.spec_from_file_location('tested_one_cell_builder', script)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        original_read = Path.read_text
        original_write = Path.write_text
        for default_encoding, input_newline in [('utf-8', '\n'), ('cp1252', '\r\n')]:
            with self.subTest(default_encoding=default_encoding), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'scripts').mkdir()
                (root / 'notebooks').mkdir()
                template = (
                    '# Editor — preserve the speaker’s meaning…\n'
                    "BUNDLE = '__SOURCE_BUNDLE__'\n"
                    "BUNDLE_SHA256 = '__SOURCE_SHA256__'\n"
                    '# __ENVIRONMENT_BOOTSTRAP__\n'
                )
                bootstrap = "LABEL = 'Narration — ready…'\n"
                for name, text in [('kaggle_one_cell_template.py', template),
                                   ('kaggle_bootstrap.py', bootstrap)]:
                    (root / 'scripts' / name).write_bytes(text.replace('\n', input_newline).encode('utf-8'))
                (root / 'ai_compose_fixture.py').write_bytes(b'VALUE = 1\n')

                def read_text(path, encoding=None, errors=None):
                    return original_read(path, encoding=encoding or default_encoding, errors=errors)

                def write_text(path, data, encoding=None, errors=None, newline=None):
                    return original_write(path, data, encoding=encoding or default_encoding,
                                          errors=errors, newline=newline if newline is not None else input_newline)

                with patch.object(builder, 'ROOT', root), patch.object(builder, 'FILES', ['ai_compose_fixture.py']), \
                     patch.object(Path, 'read_text', read_text), patch.object(Path, 'write_text', write_text), \
                     contextlib.redirect_stdout(io.StringIO()):
                    first_digest = builder.build()
                    first_python = (root / 'notebooks/sh01_full_editor_one_cell.py').read_bytes()
                    first_notebook = (root / 'notebooks/sh01_full_editor_one_cell.ipynb').read_bytes()
                    self.assertEqual(builder.build(), first_digest)
                    self.assertEqual((root / 'notebooks/sh01_full_editor_one_cell.py').read_bytes(), first_python)
                    self.assertEqual((root / 'notebooks/sh01_full_editor_one_cell.ipynb').read_bytes(), first_notebook)
                python_source = first_python.decode('utf-8')
                notebook = json.loads(first_notebook.decode('utf-8'))
                notebook_source = ''.join(notebook['cells'][0]['source'])
                self.assertEqual(python_source, notebook_source)
                self.assertIn('preserve the speaker’s meaning…', python_source)
                self.assertIn(bootstrap, python_source)
                self.assertNotIn(b'\r', first_python)
                self.assertNotIn(b'\r', first_notebook)


if __name__ == '__main__':
    unittest.main()
