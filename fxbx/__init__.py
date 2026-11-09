"""Public fxbX API. Artifact adapter internals retain their existing namespace."""
from pathlib import Path as _Path
import fxbex as _engine

_expected_engine=(_Path(__file__).resolve().parent.parent/'fxbex').resolve()
_engine_file=getattr(_engine,'__file__',None)
if not isinstance(_engine_file,str) or _Path(_engine_file).resolve().parent!=_expected_engine:
    raise ImportError(
        f'fxbX engine namespace collision: fxbex loaded from {_engine_file!r}; '
        f'expected the bundled engine at {_expected_engine}. A legacy fxbex '
        'installation may be shadowing fxbX. Reinstall this fxbX checkout '
        'non-editably in the active environment '
        '(python -m pip install --force-reinstall --no-deps PATH_TO_FXBX, without -e), '
        'or use a compatible SDK environment without the legacy fxbex distribution.'
    )

from fxbex import *
from fxbex import __all__, __version__
