"""Choose an SDK interpreter without modifying an existing environment."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .dependencies import installation_commands
from .package import PackageError
from .settings import settings

_PROBE = """
import importlib.metadata as m, json, sys
versions = {}
for name in sys.argv[1:]:
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps([list(sys.version_info[:2]), versions]))
"""


def _probe(python, names):
    try:
        result = subprocess.run([str(python), '-I', '-c', _PROBE, *names],
                                check=True, capture_output=True, text=True, timeout=5)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _interpreters():
    home = Path.home()
    roots = [home / 'miniconda3/envs', home / 'miniforge3/envs', home / '.conda/envs',
             home / '.local/share/fxbx/sdk-envs']
    roots.extend(Path(p).expanduser() for p in os.environ.get('FXBX_SDK_ENV_DIRS', '').split(os.pathsep) if p)
    paths = [Path(sys.executable), Path('/usr/bin/python3')]
    paths.extend(root / 'bin/python' for directory in roots for root in sorted(directory.glob('*')))
    registered = home / '.conda/environments.txt'
    try:
        registered_paths = registered.read_text().splitlines()
    except OSError:
        registered_paths = []
    paths.extend(Path(line.strip()) / 'bin/python' for line in registered_paths
                 if line.strip().startswith('/'))
    # venv Python symlinks may target the same binary but select different packages.
    return list(dict.fromkeys(p for p in paths if p.is_file()))


def choose_sdk(error, required, argv, stream):
    """Return a child exit code, or None to let the caller handle normal pip repair."""
    required = dict(required)
    if not {'furiosa-torch', 'furiosa-tcc'}.intersection(required):
        return None
    preferred = {(3, 10), (3, 11)}
    current = sys.version_info[:2]
    supported_host = sys.platform == 'linux' and os.uname().machine in ('x86_64', 'aarch64')
    if not supported_host:
        raise PackageError('Run this bundle on a Linux Furiosa RNGD host with its SDK installed.')
    explicit = bool(os.environ.get('FXBEX_WORKER_PYTHON') or settings().get('runtime_python'))
    if explicit and current in preferred:
        return None

    candidates = []
    base = None
    for path in _interpreters():
        info = _probe(path, ('fxbx', *required))
        if not info or tuple(info[0]) not in preferred:
            continue
        if base is None or (tuple(info[0]) == (3, 11) and base != Path(sys.executable)):
            base = path
        versions = info[1]
        if versions['fxbx'] and all(versions[k] and (v is None or versions[k] == v)
                                   for k, v in required.items()):
            candidates.append(path)

    if not candidates and current in preferred and not sys.stdin.isatty():
        return None
    if current not in preferred:
        print(f'Python {current[0]}.{current[1]} is outside the verified SDK setup '
              'for this bundle. Select Python 3.10 or 3.11.', file=stream)
    command = None
    if candidates:
        command = shlex.join(['env', f'FXBEX_WORKER_PYTHON={candidates[0]}',
                              str(candidates[0]), '-m', 'fxbx', *argv])
        print(f'Use a matching SDK environment with:\n  {command}', file=stream)
    if base is None:
        raise PackageError(f'{error} Prepare a Python 3.11 SDK environment and set FXBEX_WORKER_PYTHON.')
    # Use a new directory under the user's own configuration directory.
    directory = Path.home() / '.local/share/fxbx/sdk-envs'
    path = directory / 'sdk'
    index = 1
    while path.exists():
        path = directory / f'sdk-{index}'
        index += 1
    interpreter = path / 'bin/python'
    packages = installation_commands(required)
    from importlib.metadata import version
    setup = [[str(base), '-m', 'venv', str(path)],
             [str(interpreter), '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
              f'fxbx=={version("fxbx")}']]
    setup += [[str(interpreter), *cmd[1:4], '--only-binary=:all:', *cmd[4:]] for cmd in packages]
    if not packages:
        setup = []
    run = [str(interpreter), '-m', 'fxbx', *argv]

    def show_setup():
        for args in [*setup, ['env', f'FXBEX_WORKER_PYTHON={interpreter}', *run]] if setup else []:
            print(shlex.join(args), file=stream)

    # stderr can be redirected while stdin is still an interactive terminal.
    if not sys.stdin.isatty():
        if command is None:
            show_setup()
        raise error
    for number, candidate in enumerate(candidates, 1):
        print(f'  {number}. Use {candidate}', file=stream)
    create = len(candidates) + 1
    if setup:
        print(f'  {create}. Create and install a new environment at {path}', file=stream)
    print('  m. Show setup commands\n  q. Cancel\nSelect SDK environment [q]: ', end='', file=stream, flush=True)
    choice = sys.stdin.readline().strip().lower()
    if choice == 'm':
        show_setup()
        raise error
    if choice in {str(n) for n in range(1, len(candidates) + 1)}:
        run[0] = str(candidates[int(choice) - 1])
    elif setup and choice == str(create):
        directory.mkdir(parents=True, exist_ok=True)
        path.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}
        env['PIP_CONFIG_FILE'] = os.devnull
        for args in setup:
            subprocess.run(args, check=True, stdout=stream, stderr=stream, env=env)
    else:
        raise error
    env = dict(os.environ, FXBEX_WORKER_PYTHON=run[0])
    code = subprocess.call(run, env=env)
    return code if code >= 0 else 128 - code
