"""Repair a private editor venv while retaining Kaggle's CUDA package paths.

Uses only the standard library until pip is working. This source is inlined by
build_one_cell.py so the notebook does not need a checkout or a second download.
"""
import importlib.util
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import venv


def prepare_editor_environment(runtime, *, host_paths=None):
    """Create or repair the owned venv; never install into the host interpreter.

    Debian/Kaggle can expose dist-packages in the kernel but omit those paths
    from a venv. A local .pth file adds the actual host paths after local packages.
    The host's pip can install into a pip-less venv using its --python option.
    """
    runtime = Path(runtime).expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    root = runtime / 'venv'
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    log_path = runtime / 'setup.log'
    child_env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        child_env.pop(key, None)

    def execute(args, *, check=True):
        """Keep subprocess startup diagnostics, including sitecustomize errors."""
        result = subprocess.run([str(x) for x in args], env=child_env,
                                capture_output=True, text=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(
                f'Editor environment setup failed. Log: {log_path}\n'
                + (result.stdout + result.stderr)[-3000:])
        return result

    if not python.is_file() or not (root / 'pyvenv.cfg').is_file():
        # Do not rely on distro ensurepip, and do not clear an existing runtime.
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(root)

    probe = execute([python, '-I', '-c',
        'import sys,sysconfig,site,json; print(json.dumps({'
        '"prefix":sys.prefix,"version":list(sys.version_info[:2]),'
        '"sites":site.getsitepackages(),'
        '"purelib":sysconfig.get_path("purelib"),'
        '"platlib":sysconfig.get_path("platlib")}))'])
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    if Path(info['prefix']).resolve() != root.resolve():
        raise RuntimeError(f'Refusing to install outside the editor venv: {info["prefix"]}')
    if info['version'] != list(sys.version_info[:2]):
        raise RuntimeError('The editor venv uses a different Python version. Use a new runtime directory.')

    local_sites = []
    for value in [info['purelib'], info['platlib'], *info['sites']]:
        path = Path(value).resolve()
        if path.is_relative_to(root) and path not in local_sites:
            path.mkdir(parents=True, exist_ok=True)
            local_sites.append(path)
    if not local_sites:
        raise RuntimeError('No package directory inside the editor venv was found.')

    if host_paths is None:
        host_paths = [*site.getsitepackages(), *sys.path]
    shared = []
    for value in host_paths:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if (path.name not in ('site-packages', 'dist-packages') or not path.is_dir()
                or path.is_relative_to(root) or path in shared):
            continue
        if '\n' in str(path) or '\r' in str(path):
            raise RuntimeError('A package directory contains an invalid newline.')
        shared.append(path)
    # Also bridge distro purelib/site differences inside the venv itself.
    for local in local_sites:
        paths = [p for p in local_sites if p != local] + shared
        (local / 'sh01-kaggle-packages.pth').write_text(
            ''.join(str(p) + '\n' for p in paths), encoding='utf-8')

    health_code = (
        'import json,sys,pip,wrapt; '
        'print(json.dumps({"prefix":sys.prefix,"pip":pip.__file__,"wrapt":wrapt.__file__}))'
    )

    def healthy(result):
        if result.returncode or 'Error in sitecustomize' in result.stderr:
            return False
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
            return (Path(data['prefix']).resolve() == root and
                    all(Path(data[key]).resolve().is_relative_to(root) for key in ('pip', 'wrapt')))
        except (ValueError, IndexError, KeyError):
            return False

    health = execute([python, '-I', '-c', health_code], check=False)
    if not healthy(health):
        (runtime / 'dependencies-ready').unlink(missing_ok=True)
        spec = importlib.util.find_spec('pip')
        if spec is None or not spec.origin:
            raise RuntimeError('The notebook Python has no pip. Install pip for that interpreter first.')
        # pip's directory __main__ prepends its entire site-packages directory,
        # allowing legacy dataclasses/typing backports to shadow the stdlib.
        # Its cross-interpreter runner redirects only the pip package instead.
        runner = Path(spec.origin).parent / '__pip-runner__.py'
        if not runner.is_file():
            raise RuntimeError('Host pip needs its cross-interpreter runner (pip >=22.3).')
        # --ignore-installed makes both bootstrap packages local without removing
        # any globally installed version. --no-deps prevents unrelated changes.
        print('Repairing editor pip and startup dependency…', flush=True)
        execute([sys.executable, '-I', runner, '--python', python,
                 'install', '--disable-pip-version-check', '--no-input',
                 '--ignore-installed', '--no-deps', 'pip>=23,<27', 'wrapt>=1.16,<3'])
        health = execute([python, '-I', '-c', health_code], check=False)
        if not healthy(health):
            raise RuntimeError(f'Editor startup check still fails; see {log_path}\n'
                               + (health.stdout + health.stderr)[-3000:])
    return python, child_env
