"""Install suggestions from fxbX-owned package names and indexes."""
from importlib import metadata
import os
import re
import shlex
import subprocess
import sys

from .package import PackageError


RNGD_SETUP = 'https://developer.furiosa.ai/v2026.4.0/en/get_started/prerequisites.html'
_PYPI = frozenset({'furiosa-torch', 'furiosa-tcc', 'furiosa-kernels',
                  'furiosa-kernels-contrib', 'transformers', 'safetensors',
                  'diffusers', 'accelerate', 'onnx', 'onnx2torch', 'numpy', 'pillow', 'sentencepiece',
                  'trimesh', 'scikit-image', 'einops', 'omegaconf', 'xatlas', 'moderngl'})
_IMPORTS = {name: name for name in (*_PYPI, 'torch', 'torchvision') if '-' not in name}
_IMPORTS.update({'PIL': 'pillow', 'furiosa': 'furiosa-torch', 'furiosa.torch': 'furiosa-torch'})
_IMPORTS['skimage'] = 'scikit-image'
_VERSION = re.compile(r'[0-9][0-9A-Za-z.]*(?:\+[0-9A-Za-z.]+)?')


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class MissingPackages(PackageError):
    def __init__(self, missing):
        self.missing = missing
        names = ', '.join(f'{name}=={version}' if version else name for name, version in missing.items())
        super().__init__(f'Missing required packages in Python {sys.executable}: {names}. '
                         'Use a compatible SDK Python with fxbX installed '
                         '(set FXBEX_WORKER_PYTHON to its executable).')


def installation_commands(missing):
    commands = []
    pypi = []
    torch_version = installed_version('torch') or missing.get('torch')
    missing = dict(missing)
    if 'onnx2torch' in missing:
        if not isinstance(torch_version, str) or not _VERSION.fullmatch(torch_version):
            return []
        if installed_version('torchvision') is None:
            missing.setdefault('torchvision', None)
    for name, version in missing.items():
        if name not in _PYPI and name not in ('torch', 'torchvision'):
            return []
        if version is not None and (not isinstance(version, str) or not _VERSION.fullmatch(version)):
            return []
        requirement = f'{name}=={version}' if version else name
        if name == 'torchvision':
            if not isinstance(torch_version, str) or not _VERSION.fullmatch(torch_version) or not torch_version.endswith('+cpu'):
                return []
            commands.append([sys.executable, '-m', 'pip', 'install', '--no-input', '--index-url',
                             'https://download.pytorch.org/whl/cpu', f'torch=={torch_version}', requirement])
        elif name == 'torch':
            if version is not None and '+' in version and not version.endswith('+cpu'):
                return []
            commands.append([sys.executable, '-m', 'pip', 'install', '--no-input', '--index-url',
                             'https://download.pytorch.org/whl/cpu', requirement])
        else:
            pypi.append(requirement)
    if pypi:
        if 'onnx2torch' in missing:
            pypi.append(f'torch=={torch_version}')
        # Resolve adapter dependencies together with the CLI's own Hub requirement.
        if 'transformers' in missing:
            pypi.append('huggingface_hub>=0.24,<2')
        commands.append([sys.executable, '-m', 'pip', 'install', '--no-input', '--index-url',
                         'https://pypi.org/simple', *pypi])
    return commands


def missing_import(error):
    name = _IMPORTS.get(error.name)
    if name and installed_version(name) is None:
        return MissingPackages({name: None})
    return None


def offer_install(error, stream):
    commands = installation_commands(error.missing)
    if not commands:
        return False
    for command in commands:
        print(shlex.join(command), file=stream)
    if not sys.stdin.isatty():
        return False
    print(error, file=stream)
    print('Install the missing Python packages above? [y/N] ', end='', file=stream, flush=True)
    if sys.stdin.readline().strip().lower() not in ('y', 'yes'):
        return False
    env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
    env['PIP_CONFIG_FILE'] = os.devnull
    for command in commands:
        # Leave stdout available for the original command's JSON result.
        subprocess.run(command, check=True, stdout=stream, stderr=stream, env=env)
    return True
