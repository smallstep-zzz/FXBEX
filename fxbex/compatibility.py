"""Report version drift while leaving executable checks to the SDK loader."""
from importlib import metadata
import warnings


def version_notice(name, expected, installed):
    return (f'SDK mismatch {name}: package={expected}, installed={installed}; '
            'attempting execution with this unverified combination')


def check_edf_versions(packages, error):
    # These EDF loaders consume compiled code. TCC is optional build provenance.
    for name in ('torch', 'furiosa-torch', 'furiosa-tcc'):
        if name not in packages:
            continue
        expected = packages[name]
        if not isinstance(expected, str) or not expected:
            raise error(f'Invalid SDK version metadata for {name}')
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            if name == 'furiosa-tcc':
                continue
            raise error(f'Missing required runtime package: {name}') from exc
        if installed != expected:
            warnings.warn(version_notice(name, expected, installed), RuntimeWarning, stacklevel=3)
