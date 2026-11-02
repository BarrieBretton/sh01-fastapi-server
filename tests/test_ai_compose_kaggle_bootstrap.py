"""Offline bootstrap regression tests using real subprocesses and pip.

The tiny wrapt fixture tests package installation/location, not wrapt behavior.
No model downloads, GPU, network, or host-package mutations are needed.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import venv
import zipfile

from scripts.kaggle_bootstrap import prepare_editor_environment


class KaggleBootstrapTests(unittest.TestCase):
    def setUp(self):
        try:
            import ensurepip
        except ImportError:
            self.skipTest('Offline bootstrap test needs an ensurepip wheel; runtime bootstrap does not.')
        wheels = list((Path(ensurepip.__file__).parent / '_bundled').glob('pip-*.whl'))
        if not wheels:
            self.skipTest('No bundled pip wheel for the offline bootstrap test.')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'runtime'
        self.wheels = self.root / 'wheels'
        self.wheels.mkdir()
        pip_wheel = wheels[0]
        shutil.copy2(pip_wheel, self.wheels)
        with zipfile.ZipFile(self.wheels / 'wrapt-2.0.0-py3-none-any.whl', 'w') as out:
            files = {
                'wrapt/__init__.py': '__version__ = "2.0.0"\n',
                'wrapt-2.0.0.dist-info/METADATA': 'Metadata-Version: 2.1\nName: wrapt\nVersion: 2.0.0\n',
                'wrapt-2.0.0.dist-info/WHEEL': 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
            }
            for name, data in files.items():
                out.writestr(name, data)
            out.writestr('wrapt-2.0.0.dist-info/RECORD',
                         ''.join(f'{name},,\n' for name in files) + 'wrapt-2.0.0.dist-info/RECORD,,\n')
        self.env = mock.patch.dict(os.environ, {'PIP_NO_INDEX': '1', 'PIP_FIND_LINKS': str(self.wheels),
                                              'PIP_CONFIG_FILE': os.devnull})
        self.env.start()
        self.addCleanup(self.env.stop)

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_editor_environment(self.runtime, **kwargs)

    def query(self, python, code, env):
        result = subprocess.run([str(python), '-c', code], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Error in sitecustomize', result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_fresh_bootstrap_installs_locally_and_rerun_is_idempotent(self):
        import pip
        original = hashlib.sha256(Path(pip.__file__).read_bytes()).hexdigest()
        python, env = self.prepare()
        data = self.query(python, 'import sys,pip,wrapt,json; print(json.dumps('
                          '[sys.prefix,pip.__file__,wrapt.__file__]))', env)
        for path in data:
            self.assertTrue(Path(path).resolve().is_relative_to(self.runtime / 'venv'))
        with mock.patch('scripts.kaggle_bootstrap.subprocess.run', wraps=subprocess.run) as run:
            self.prepare()
        self.assertFalse(any('install' in call.args[0] for call in run.call_args_list))
        self.assertEqual(hashlib.sha256(Path(pip.__file__).read_bytes()).hexdigest(), original)

    def test_repairs_existing_pipless_venv_and_missing_startup_dependency(self):
        root = self.runtime / 'venv'
        venv.EnvBuilder(with_pip=False).create(root)
        host = self.root / 'custom-host' / 'dist-packages'
        host.mkdir(parents=True)
        (host / 'sitecustomize.py').write_text('import wrapt\n')
        (host / 'sh01_probe.py').write_text('value = "host"\n')
        (self.runtime / 'dependencies-ready').write_text('stale')
        python, env = self.prepare(host_paths=[str(host)])
        self.assertFalse((self.runtime / 'dependencies-ready').exists())
        data = self.query(python, 'import sh01_probe,sitecustomize,json; print(json.dumps('
                          '[sh01_probe.value,sitecustomize.__file__]))', env)
        self.assertEqual(data, ['host', str(host / 'sitecustomize.py')])
        self.assertIn("No module named 'wrapt'", (self.runtime / 'setup.log').read_text())
        local = self.query(python, 'import sysconfig,json; print(json.dumps(sysconfig.get_path("purelib")))', env)
        (Path(local) / 'sh01_probe.py').write_text('value = "local"\n')
        data = self.query(python, 'import sh01_probe,json; print(json.dumps(sh01_probe.value))', env)
        self.assertEqual(data, 'local')

    def test_failed_install_invalidates_ready_stamp_and_preserves_job_files(self):
        root = self.runtime / 'venv'
        venv.EnvBuilder(with_pip=False).create(root)
        stamp = self.runtime / 'dependencies-ready'
        stamp.write_text('stale')
        job = self.runtime / 'saved-job.json'
        job.write_text('{"status":"complete"}')
        real_run = subprocess.run

        def fail_install(args, **kwargs):
            if 'install' in args:
                return subprocess.CompletedProcess(args, 1, '', 'simulated offline installer failure')
            return real_run(args, **kwargs)

        with mock.patch('scripts.kaggle_bootstrap.subprocess.run', side_effect=fail_install):
            with self.assertRaisesRegex(RuntimeError, 'simulated offline installer failure'):
                self.prepare(host_paths=[])
        self.assertFalse(stamp.exists())
        self.assertEqual(json.loads(job.read_text()), {'status': 'complete'})

    def test_refuses_install_into_host_prefix(self):
        root = self.runtime / 'venv'
        venv.EnvBuilder(with_pip=False).create(root)
        bogus = json.dumps({'prefix':sys.prefix, 'version':list(sys.version_info[:2]),
                            'sites':[], 'purelib':sys.prefix, 'platlib':sys.prefix})
        with mock.patch('scripts.kaggle_bootstrap.subprocess.run',
                        return_value=subprocess.CompletedProcess([], 0, bogus, '')) as run:
            with self.assertRaisesRegex(RuntimeError, 'outside the editor venv'):
                self.prepare()
        self.assertEqual(run.call_count, 1)

    def test_inherited_python_and_pip_overrides_are_removed(self):
        with mock.patch.dict(os.environ, {'PYTHONPATH':'/wrong', 'PIP_TARGET':'/wrong',
                                         'PIP_PREFIX':'/wrong', 'PIP_USER':'1'}):
            _, env = self.prepare()
        for key in ('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
            self.assertNotIn(key, env)


if __name__ == '__main__':
    unittest.main()
