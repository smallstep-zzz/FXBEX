"""Write weightless FXB archives from an existing native manifest and EDFs.

This preserves a native producer's execution contract; it cannot translate an
FXBEX Runnable graph into Furiosa-LLM kernels or invent an architecture fingerprint.
"""
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

PRODUCER_MEMBER = 'fxbex.provenance.json'
MAX_JSON_SIZE = 4 * 1024**2


class NativeFXBError(ValueError):
    """The native archive contract or official reader gate failed."""


def _json(data):
    if len(data) > MAX_JSON_SIZE:
        raise NativeFXBError('Native JSON exceeds size limit')
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise NativeFXBError(f'Invalid native JSON: {exc}') from exc


def _name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute()
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise NativeFXBError(f'Unsafe native member: {name!r}')
    return name


def _contract(manifest):
    required = {'format_version', 'uuid', 'created_at', 'versions', 'model',
                'parallel_config', 'kernels'}
    if (not isinstance(manifest, dict) or not required.issubset(manifest)
            or type(manifest['format_version']) is not int or manifest['format_version'] != 2):
        raise NativeFXBError('EXPORT_BRIDGE_GAP: expected an existing official native FXB v2 manifest; FXBEX/Runnable conversion is unavailable')
    if not isinstance(manifest['model'], dict) or not manifest['model'].get('architecture'):
        raise NativeFXBError('Native manifest must preserve its real architecture fingerprint')
    kernels = manifest['kernels']
    if not isinstance(kernels, dict) or not kernels:
        raise NativeFXBError('Native manifest has no kernels')
    files, count = set(), 0
    for entries in kernels.values():
        if not isinstance(entries, list) or not entries:
            raise NativeFXBError('Native kernel entries must be nonempty lists')
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get('bucket'), dict):
                raise NativeFXBError('Native kernel entry requires file and bucket')
            name = _name(entry.get('file'))
            if not name.startswith('edfs/') or not name.endswith('.edf'):
                raise NativeFXBError('Native kernels must reference edfs/*.edf; weights and Runnable files are excluded')
            files.add(name)
            count += 1
    return files, count


def _official(path):
    try:
        from furiosa.llm_native import fxb
    except ImportError as exc:
        raise NativeFXBError('Official FXB reader unavailable; use the matching installed Furiosa SDK') from exc
    for gate in ('show', 'inspect'):
        try:
            # These public readers parse metadata and EDF signatures, not inference.
            getattr(fxb, gate)(str(path))
        except Exception as exc:
            raise NativeFXBError(f'Official fxb {gate} rejected archive: {exc}') from exc


def validate_native(path, *, verify_official=True):
    """Check structure and optionally official readers; never claim inference."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            members = {}
            for info in archive.infolist():
                name = _name(info.filename)
                if name in members:
                    raise NativeFXBError(f'Duplicate native member: {name}')
                mode = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or mode not in (0, stat.S_IFREG):
                    raise NativeFXBError(f'Unsupported native member: {name}')
                if info.flag_bits & 1:
                    raise NativeFXBError('Encrypted native members are unsupported')
                members[name] = info
            if 'manifest.json' not in members:
                raise NativeFXBError('Missing native manifest.json')
            if members['manifest.json'].file_size > MAX_JSON_SIZE:
                raise NativeFXBError('Native JSON exceeds size limit')
            manifest = _json(archive.read('manifest.json'))
            files, count = _contract(manifest)
            expected = files | {'manifest.json'}
            producer = None
            if PRODUCER_MEMBER in members:
                expected.add(PRODUCER_MEMBER)
                if members[PRODUCER_MEMBER].file_size > MAX_JSON_SIZE:
                    raise NativeFXBError('Native JSON exceeds size limit')
                producer = _json(archive.read(PRODUCER_MEMBER))
                if not isinstance(producer, dict) or producer.get('producer') != 'fxbex':
                    raise NativeFXBError('Invalid FXBEX producer metadata')
            if set(members) != expected:
                raise NativeFXBError('Native archive members do not match manifest EDF references and producer metadata')
            if any(members[name].file_size == 0 for name in files):
                raise NativeFXBError('Empty EDF kernel')
            bad = archive.testzip()
            if bad:
                raise NativeFXBError(f'Native archive CRC mismatch: {bad}')
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
        raise NativeFXBError(f'Invalid native archive: {exc}') from exc
    if verify_official:
        _official(path)
    reader = 'passed' if verify_official else 'not_run'
    return {'schema': 'fxbex.native-fxb.validation.v1', 'path': str(path),
            'format': 'FXB', 'format_version': 2, 'weightless': True,
            'kernel_entries': count, 'edf_files': len(files), 'producer_metadata': producer,
            'gates': {'archive': 'passed', 'official_show': reader, 'official_inspect': reader,
                      'internal_metadata_reader': reader if producer else 'not_applicable',
                      'model_compatibility': 'not_run', 'runtime_load': 'not_run',
                      'npu_execution': 'not_run'}}


def build_native(source_dir, output, *, verify_official=True, producer_metadata=None):
    """Archive native manifest+EDFs unchanged, add producer info, never overwrite.

    ``producer_metadata`` is experimental FXBEX metadata in a separate ZIP entry,
    not part of the official manifest or fingerprint. Reader acceptance does not
    establish that the full runtime supports this extension.
    """
    from . import __version__
    source, output = Path(source_dir).resolve(), Path(output).absolute()
    if not source.is_dir():
        raise NativeFXBError('Source must be an extracted native artifact directory')
    if output.suffix != '.fxb':
        raise NativeFXBError('Native output must use .fxb')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest_path = source / 'manifest.json'
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise NativeFXBError('Missing regular native manifest.json')
    if manifest_path.stat().st_size > MAX_JSON_SIZE:
        raise NativeFXBError('Native JSON exceeds size limit')
    manifest_bytes = manifest_path.read_bytes()
    files, _ = _contract(_json(manifest_bytes))
    for name in files:
        path = source / name
        if (not path.is_file() or not path.stat().st_size
                or any((source.joinpath(*PurePosixPath(name).parts[:i])).is_symlink()
                       for i in range(1, len(PurePosixPath(name).parts) + 1))):
            raise NativeFXBError(f'Missing regular native EDF or symlink in path: {name}')
    metadata = dict(producer_metadata or {})
    metadata.update({'schema': 'fxbex.producer.v1', 'producer': 'fxbex',
                     'version': __version__, 'operation': 'archive_existing_native_artifact'})
    try:
        metadata_bytes = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError) as exc:
        raise NativeFXBError(f'Invalid producer metadata: {exc}') from exc
    _json(metadata_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native-fxb-', dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr('manifest.json', manifest_bytes)
            for name in sorted(files):
                archive.write(source / name, name)
            archive.writestr(PRODUCER_MEMBER, metadata_bytes)
        report = validate_native(staged, verify_official=verify_official)
        # A hard link publishes atomically and fails if another producer won.
        os.link(staged, output)
    report['path'] = str(output)
    return report
