"""Validate native FXB archives and SDK reader compatibility."""
import json
from pathlib import Path, PurePosixPath
import stat
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
        raise NativeFXBError('EXPORT_BRIDGE_GAP: expected a native FXB v2 manifest and genuine EDFs; serialized Runnable conversion is unavailable')
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
        # Torch and LLM can live in separate SDK Python environments on the same host.
        import shutil,subprocess
        cli=shutil.which('fxb')
        if cli is None:
            raise NativeFXBError('Official FXB reader unavailable; install matching SDK or expose its fxb CLI') from exc
        for gate in ('show','inspect'):
            result=subprocess.run([cli,gate,str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            if result.returncode:
                raise NativeFXBError(f'Official fxb {gate} rejected archive: {result.stderr[-1500:]}')
        return
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
