"""User-owned process settings, never read from an untrusted model package."""
import json
import os
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


def runs_path():
    return Path(settings().get('reports_dir',str(cache_root()/'runs'))).expanduser().resolve()
