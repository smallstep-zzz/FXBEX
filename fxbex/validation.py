"""Integrity and static deployment checks; never load a kernel or allocate an NPU."""
import json

from .package import extract, validate as validate_integrity


def validate(path, *, integrity_only=False, weights_dir=None):
    if integrity_only:
        manifest = validate_integrity(path)
        return {'valid': True, 'integrity_verified': True, 'statically_ready': None,
                'inference_validated': False, 'model': manifest['model'],
                'checks': {'readiness': 'not executed', 'inference': 'not executed; no NPU execution evidence'}}
    from . import runtime
    from .memory import analyze, summarize
    checks = {'integrity': 'passed', 'inference': 'not executed; no NPU execution evidence'}
    errors = []
    # extract validates the archive once and bounds the configuration sizes.
    with extract(path) as root:
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, check in (
            ('compatibility', lambda: runtime.check_compatibility(manifest)),
            ('memory_plan', lambda: summarize(analyze(root))),
            ('external_weights', lambda: _weights(root, weights_dir)),
            ('adapter_preflight', lambda: _adapter_preflight(runtime, manifest, root, weights_dir)),
        ):
            try:
                detail = check()
                checks[name] = detail or 'passed'
            except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError, AttributeError) as exc:
                checks[name] = 'failed'
                errors.append(f'{name}: {exc}')
        environment = runtime.doctor()
        device = environment.get('device', {})
        device_ready = device.get('returncode') == 0 and 'rngd' in device.get('status', '').lower()
        checks['device_detection'] = 'passed' if device_ready else 'failed'
        if not device_ready:
            errors.append('device_detection: furiosa-smi status did not report an available RNGD device')
    return {'valid': not errors, 'integrity_verified': True, 'statically_ready': not errors,
            'inference_validated': False, 'model': manifest['model'], 'checks': checks,
            'environment': environment, 'errors': errors,
            'note': 'Static readiness does not prove kernel loading or inference execution.'}


def _weights(root, weights_dir):
    import hashlib
    from pathlib import Path
    from .package import CHUNK, PackageError
    index = json.loads((root / 'weights_index.json').read_text())
    for item in index['weights']:
        if 'external_path' not in item:
            continue  # Packaged bytes were already hashed by extract.
        original = Path(item['external_path'])
        name = item.get('name', original.name)
        if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..') or '\\' in name:
            raise PackageError('External weight name must be a basename')
        path = Path(weights_dir).resolve() / name if weights_dir else original
        if not path.is_absolute():
            raise PackageError('Relative external weights require --weights-dir')
        if not path.is_file():
            raise PackageError(f'Missing external weight: {path}')
        expected = item.get('sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise PackageError(f'External weight requires SHA-256: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(CHUNK), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise PackageError(f'External weight checksum mismatch: {path}')


def _adapter_preflight(runtime, manifest, root, weights_dir):
    from .bundled import activate
    if manifest['runtime']['adapter'] != 'runnable_dag':
        with activate(root,manifest) as bundled:
            adapter = bundled or runtime.resolve_adapter(manifest['runtime']['adapter'])
            adapter.preflight(root, weights_dir=weights_dir)
        return
    from .package import PackageError
    plan = json.loads((root / 'execution_plan.json').read_text())
    config = json.loads((root / 'runtime_config.json').read_text())
    index = json.loads((root / 'weights_index.json').read_text())
    if plan.get('format') != 'runnable-dag-v1' or not isinstance(plan.get('steps'), list):
        raise PackageError('Expected runnable-dag-v1 plan with ordered steps')
    if not isinstance(config.get('inputs'), dict) or not isinstance(config.get('outputs'), list):
        raise PackageError('DAG runtime config requires inputs object and outputs list')
    known = set(config['inputs'])
    unknown_weights = False
    for item in index['weights']:
        tensors = item.get('tensors')
        if tensors is None:
            unknown_weights = True
            continue
        if not isinstance(tensors, (list, dict)):
            raise PackageError('DAG weight tensors must be a list or object')
        for name in tensors:
            if not isinstance(name, str) or name in known:
                raise PackageError('Duplicate or invalid DAG weight binding')
            known.add(name)
    ids = set()
    for step in plan['steps']:
        if not isinstance(step, dict) or not isinstance(step.get('id'), str) or step['id'] in ids:
            raise PackageError('DAG steps require unique string ids')
        ids.add(step['id'])
        if not isinstance(step.get('inputs'), list) or not isinstance(step.get('outputs'), list):
            raise PackageError('DAG step inputs and outputs must be lists')
        if not unknown_weights and any(name not in known for name in step['inputs']):
            raise PackageError('Unbound or forward DAG input')
        outputs = step['outputs']
        if any(not isinstance(name, str) or name in known for name in outputs) or len(set(outputs)) != len(outputs):
            raise PackageError('Duplicate or invalid DAG tensor binding')
        if step.get('artifact') not in manifest['files']:
            raise PackageError('DAG artifact is missing from package integrity map')
        known.update(outputs)
    if not unknown_weights and any(name not in known for name in config['outputs']):
        raise PackageError('Unbound DAG output')
    if unknown_weights:
        return 'artifacts checked; weight tensor bindings require runtime inspection (undeclared tensors)'
