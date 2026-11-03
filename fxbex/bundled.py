"""Explicit trusted Python runtime bundles; not a sandbox for untrusted packages."""
from contextlib import contextmanager
import importlib
import re
import sys
from .package import PackageError


def specification(manifest):
    spec=manifest['runtime'].get('bundled_python')
    if spec is None:return None
    if not isinstance(spec,dict) or spec.get('root')!='python':raise PackageError('Expected bundled_python root=python')
    entry=spec.get('entrypoint','')
    if not isinstance(entry,str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*',entry):
        raise PackageError('Invalid bundled adapter entrypoint')
    packages=spec.get('packages')
    if not isinstance(packages,list) or not packages or any(not isinstance(p,str) or not re.fullmatch(r'[A-Za-z_]\w*',p) for p in packages):
        raise PackageError('Invalid bundled top-level packages')
    if len(set(packages))!=len(packages) or entry.split('.')[0].split(':')[0] not in packages or 'fxbex' in packages:
        raise PackageError('Invalid bundled package ownership')
    for p in packages:
        if f'python/{p}/__init__.py' not in manifest['files']:raise PackageError(f'Missing bundled package {p}')
    module=entry.split(':')[0].replace('.','/')
    if f'python/{module}.py' not in manifest['files']:raise PackageError('Missing bundled entrypoint source')
    return spec


@contextmanager
def activate(root,manifest):
    spec=specification(manifest)
    if spec is None:
        yield None;return
    names=spec['packages']
    def owned(n):return any(n==p or n.startswith(p+'.') for p in names)
    if any(owned(n) for n in sys.modules):
        raise PackageError('Bundled runtime conflicts with imported modules; use a fresh FXBEX worker')
    directory=str(root/'python');sys.path.insert(0,directory)
    try:
        module,attribute=spec['entrypoint'].split(':')
        yield getattr(importlib.import_module(module),attribute)
    finally:
        if directory in sys.path:sys.path.remove(directory)
        for n in list(sys.modules):
            if owned(n):del sys.modules[n]
        importlib.invalidate_caches()
