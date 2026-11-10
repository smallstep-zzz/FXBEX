"""User-owned process settings, never read from an untrusted model package."""
import json
import math
import os
import sys
import threading
import contextlib
import fcntl
from pathlib import Path


def config_path():
    return Path(os.environ.get('FXBEX_CONFIG',Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'fxbex/config.json'))


def settings():
    path=config_path()
    value=json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(value,dict):raise ValueError('FXBEX config must be a JSON object')
    return value


def cache_root():
    return Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'fxbex'


def lock_path():
    return Path(os.environ.get('FXBEX_NPU_LOCK',settings().get('npu_lock',str(cache_root()/'npu.lock')))).expanduser().resolve()


def inherited_lease_fd():
    """Return the validated host lease FD for an explicit subprocess pass_fds."""
    inherited=os.environ.get('FXBEX_NPU_LOCK_FD')
    if inherited is None:return None
    try:
        fd=int(inherited);actual,expected=os.fstat(fd),lock_path().stat()
    except (ValueError,OSError) as error:
        raise RuntimeError('Invalid inherited NPU lease descriptor') from error
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
        raise RuntimeError('Inherited NPU lease does not match canonical lock')
    return fd


@contextlib.contextmanager
def npu_lock():
    """Acquire or adopt a scheduler lease; never unlock the parent's file description."""
    path=lock_path();path.parent.mkdir(parents=True,exist_ok=True)
    inherited=os.environ.get('FXBEX_NPU_LOCK_FD')
    if inherited is not None:
        fd=os.dup(int(inherited))
        try:
            actual,expected=os.fstat(fd),path.stat()
            if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
                raise RuntimeError('Inherited NPU lease does not match canonical lock')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield fd
        finally:os.close(fd)
    else:
        with path.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield stream.fileno()


def runs_path():
    return Path(settings().get('reports_dir',str(cache_root()/'runs'))).expanduser().resolve()


def runtime_python():
    """Host-owned SDK interpreter; never selected by executable package metadata."""
    value=os.environ.get('FXBEX_WORKER_PYTHON') or settings().get('runtime_python') or sys.executable
    if not isinstance(value,str) or not Path(value).is_absolute() or not Path(value).is_file():
        raise ValueError('FXBEX runtime_python must be an existing absolute interpreter path')
    return str(Path(value))


def worker_timeouts():
    """Host-owned finite waits; packages cannot disable worker deadlines."""
    result={'load':600.,'request':300.,'unload':30.,'terminate':15.,'kill':5.}
    configured=settings().get('worker_timeouts',{})
    if not isinstance(configured,dict) or set(configured)-set(result):
        raise ValueError('worker_timeouts must contain only load/request/unload/terminate/kill seconds')
    for name,value in configured.items():
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=threading.TIMEOUT_MAX/2:
            raise ValueError('worker_timeouts values must be finite positive seconds: '+name)
        result[name]=float(value)
    return result
