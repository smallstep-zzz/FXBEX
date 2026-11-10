"""Model-independent FXB_EX archives; this is not Furiosa's official FXB format."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

REQUIRED = ('execution_plan.json', 'weights_index.json', 'memory_plan.json', 'runtime_config.json')
DEFAULT_MAX_TOTAL_SIZE = 64 * 1024**3
DEFAULT_MAX_JSON_SIZE = 16 * 1024**2
CHUNK = 1024**2


class PackageError(ValueError):
    """An invalid or unsafe FXB_EX package."""


def _name(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or ':' in name or name.startswith('/')
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise PackageError(f'Unsafe member path: {name!r}')
    return name


def _json(data, label):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PackageError(f'Duplicate JSON key in {label}: {key}')
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise PackageError(f'Invalid JSON in {label}: {exc}') from exc
    if not isinstance(value, dict):
        raise PackageError(f'{label} must be a JSON object')
    return value


def _manifest(manifest):
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest.get('format') != 'FXB_EX':
        raise PackageError('Expected FXB_EX schema_version 1')
    if (not isinstance(manifest.get('model'), dict)
            or not isinstance(manifest['model'].get('name'), str) or not manifest['model']['name']):
        raise PackageError('manifest.model.name is required')
    if not isinstance(manifest.get('compatibility'), dict):
        raise PackageError('manifest.compatibility must be an object')
    runtime = manifest.get('runtime')
    if not isinstance(runtime, dict) or not isinstance(runtime.get('adapter'), str) or not runtime['adapter']:
        raise PackageError('manifest.runtime.adapter is required')
    if not isinstance(manifest.get('standalone'), bool):
        raise PackageError('manifest.standalone must be boolean')
    files = manifest.get('files')
    if not isinstance(files, dict) or not set(REQUIRED).issubset(files):
        raise PackageError('Integrity map must include every required configuration')
    for name, item in files.items():
        _name(name)
        if name == 'manifest.json' or not isinstance(item, dict):
            raise PackageError(f'Invalid integrity entry: {name}')
        digest = item.get('sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or type(item.get('size')) is not int or item['size'] < 0):
            raise PackageError(f'Invalid checksum or size: {name}')
    from .optimization import specification
    specification(manifest)


def _weights(index, manifest):
    weights = index.get('weights')
    if not isinstance(weights, list):
        raise PackageError('weights_index.weights must be a list')
    for weight in weights:
        if not isinstance(weight, dict) or ('path' in weight) == ('external_path' in weight):
            raise PackageError('Each weight needs exactly one path or external_path')
        if 'path' in weight:
            if _name(weight['path']) not in manifest['files']:
                raise PackageError(f'Missing packaged weight: {weight["path"]}')
        elif manifest['standalone']:
            raise PackageError('Standalone packages cannot reference external weights')
        elif not isinstance(weight['external_path'], str) or not weight['external_path']:
            raise PackageError('external_path must be a nonempty string')


def _validate(archive, max_total_size, max_json_size, *, verify_hashes=True):
    if max_total_size <= 0 or max_json_size <= 0:
        raise ValueError('Size limits must be positive')
    members = {}
    total = 0
    for info in archive.infolist():
        name = _name(info.filename)
        mode = info.external_attr >> 16
        if name in members or info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
            raise PackageError(f'Duplicate or unsupported member: {name}')
        if info.flag_bits & 1:
            raise PackageError('Encrypted members are unsupported')
        total += info.file_size
        if total > max_total_size:
            raise PackageError('Package exceeds total uncompressed size limit')
        members[name] = info
    for name in members:
        if any(str(parent) in members for parent in PurePosixPath(name).parents if str(parent) != '.'):
            raise PackageError(f'Member conflicts with parent file: {name}')
    if 'manifest.json' not in members:
        raise PackageError('Missing required member: manifest.json')
    if members['manifest.json'].file_size > max_json_size:
        raise PackageError('JSON exceeds size limit: manifest.json')
    manifest = _json(archive.read('manifest.json'), 'manifest.json')
    if type(manifest.get('format_version')) is int and manifest['format_version'] == 2 and manifest.get('format') != 'FXB_EX':
        raise PackageError(
            'NATIVE_FXB_REQUIRES_EXPLICIT_LOADER: this is a native FXB v2, not an FXB_EX package. '
            'For a TCLMachine execution plan use fxbx.GraphFXB(path, weights=...). '
            'For an official LLM FXB use furiosa_llm.LLM(model_dir, fxb=path). '
            'fxbx.Model/LLM and the package CLI accept FXB_EX; do not rename the file to convert it.')
    for name in REQUIRED:
        if name not in members:
            raise PackageError(f'Missing required member: {name}')
        if members[name].file_size > max_json_size:
            raise PackageError(f'JSON exceeds size limit: {name}')
    _manifest(manifest)
    if set(members) != {'manifest.json', *manifest['files']}:
        raise PackageError('Archive members do not match manifest integrity map')
    for name, entry in manifest['files'].items():
        if members[name].file_size != entry['size']:
            raise PackageError(f'Size mismatch: {name}')
        if verify_hashes:
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                while chunk := stream.read(CHUNK):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise PackageError(f'Checksum mismatch: {name}')
    from .bundled import specification
    specification(manifest)
    configs = {name: _json(archive.read(name), name) for name in REQUIRED}
    _weights(configs['weights_index.json'], manifest)
    return manifest


def validate(path, *, max_total_size=DEFAULT_MAX_TOTAL_SIZE, max_json_size=DEFAULT_MAX_JSON_SIZE):
    """Verify member safety, bounded sizes, required JSONs and all SHA-256 checksums.

    External weight existence and runtime/NPU compatibility are not checked.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return _validate(archive, max_total_size, max_json_size)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise PackageError(f'Unreadable archive: {exc}') from exc


def inspect(path, **limits):
    """Return the manifest only after full archive validation."""
    return validate(path, **limits)


@contextmanager
def extract(path, *, max_total_size=DEFAULT_MAX_TOTAL_SIZE, max_json_size=DEFAULT_MAX_JSON_SIZE):
    """Yield a validated temporary package directory, deleting it on exit.

    One open archive is used for validation and extraction to avoid path replacement.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = _validate(archive, max_total_size, max_json_size, verify_hashes=False)
            with tempfile.TemporaryDirectory(prefix='fxbex-') as directory:
                root = Path(directory)
                for info in archive.infolist():
                    target = root.joinpath(*PurePosixPath(info.filename).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open(info) as source, target.open('xb') as dest:
                        while chunk := source.read(CHUNK):
                            dest.write(chunk)
                            digest.update(chunk)
                    if info.filename != 'manifest.json' and digest.hexdigest() != manifest['files'][info.filename]['sha256']:
                        raise PackageError(f'Checksum mismatch: {info.filename}')
                yield root
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise PackageError(f'Unreadable archive: {exc}') from exc


def build(source, output, standalone=False, *, max_total_size=DEFAULT_MAX_TOTAL_SIZE, max_json_size=DEFAULT_MAX_JSON_SIZE, source_url=None, source_revision=None):
    """Build atomically without overwriting; source is an unpacked package tree.

    Source manifest supplies model, compatibility and runtime. Integrity entries and
    standalone are generated. weights_index uses {"weights": [{"path": "weights/a"}]}
    for internal weights or {"external_path": "/deployment/a"} for external weights.
    In standalone mode external files are streamed into weights/ and their index
    entries are rewritten in the archive only. Relative external paths resolve
    against source; explicit external symlinks are resolved (e.g. HF cache blobs).
    """
    source = Path(source).resolve()
    output = Path(output).absolute()
    if not source.is_dir():
        raise PackageError('Source must be a directory')
    if output.resolve().is_relative_to(source):
        raise PackageError('Output must be outside the source directory')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest_path = source / 'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PackageError('Source requires a regular manifest.json')
    if manifest_path.stat().st_size > max_json_size:
        raise PackageError('Manifest exceeds JSON size limit')
    manifest = _json(manifest_path.read_bytes(), 'manifest.json')
    from .provenance import source_metadata
    manifest['provenance']=source_metadata(source_url,source_revision,manifest.get('provenance'))
    from .buildbundle import validate as validate_bundle
    validate_bundle(source,manifest)
    manifest.update(schema_version=1, format='FXB_EX', standalone=bool(standalone), files={})
    inputs = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise PackageError(f'Source symlinks are forbidden: {path}')
        if path.is_dir():
            continue
        if not path.is_file():
            raise PackageError(f'Source must contain regular files: {path}')
        name = _name(path.relative_to(source).as_posix())
        if name != 'manifest.json':
            inputs[name] = (path, None)
    if standalone:
        index_path = source / 'weights_index.json'
        if not index_path.is_file() or index_path.stat().st_size > max_json_size:
            raise PackageError('Missing or oversized weights_index.json')
        index = _json(index_path.read_bytes(), 'weights_index.json')
        # Validate the original references before replacing external entries.
        _weights(index, {'standalone': False, 'files': inputs})
        for weight in index['weights']:
            if 'external_path' not in weight:
                continue
            external = Path(weight['external_path'])
            basename = weight.get('name', external.name)
            _name(basename)
            if '/' in basename:
                raise PackageError(f'Weight name must be a basename: {basename}')
            name = 'weights/' + basename
            if name in inputs:
                raise PackageError(f'Weight path collision: {name}')
            external = (external if external.is_absolute() else source / external).resolve()
            if not external.is_file():
                raise PackageError(f'Missing external weight file: {external}')
            inputs[name] = (external, weight.get('sha256'))
            del weight['external_path']
            weight['path'] = name
        inputs['weights_index.json'] = (json.dumps(index, sort_keys=True).encode(), None)
    if manifest['runtime']['adapter'] == 'runnable_dag':
        from .memory import dag_plan
        def document(name):
            item = inputs[name][0]
            if isinstance(item, bytes):
                data = item
            else:
                if item.stat().st_size > max_json_size:
                    raise PackageError(f'Oversized memory metadata: {name}')
                data = item.read_bytes()
            return _json(data, name)
        memory = dag_plan(document('execution_plan.json'), document('runtime_config.json'),
                          document('weights_index.json'), document('memory_plan.json'))
        data = json.dumps(memory, sort_keys=True).encode()
        if len(data) > max_json_size:
            raise PackageError('Generated memory plan exceeds JSON size limit')
        inputs['memory_plan.json'] = (data, None)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.fxbex-', dir=output.parent)
    os.close(fd)
    try:
        total = 0
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, (item, expected_digest) in inputs.items():
                digest = hashlib.sha256()
                size = 0
                with (io.BytesIO(item) if isinstance(item, bytes) else item.open('rb')) as stream:
                    with archive.open(name, 'w', force_zip64=True) as dest:
                        while chunk := stream.read(CHUNK):
                            size += len(chunk)
                            total += len(chunk)
                            if total > max_total_size:
                                raise PackageError('Source exceeds total size limit')
                            dest.write(chunk)
                            digest.update(chunk)
                if expected_digest is not None and digest.hexdigest() != expected_digest:
                    raise PackageError(f'External weight checksum mismatch: {name}')
                manifest['files'][name] = {'sha256': digest.hexdigest(), 'size': size}
            archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        validate(temporary, max_total_size=max_total_size, max_json_size=max_json_size)
        os.link(temporary, output)  # Atomic publication with no overwrite.
        return output
    finally:
        os.unlink(temporary)
