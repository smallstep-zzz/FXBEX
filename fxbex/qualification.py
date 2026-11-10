"""Check producer qualification bindings without implementing model semantics.

Receipts are trusted-host data, not signatures or executable package metadata.
Full artifact hashes are checked once per load, never for each inference call.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from .native_fxb import MAX_JSON_SIZE

SCHEMA = 'tclmachine.qualification-receipt.v1'


class QualificationError(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise QualificationError('Duplicate qualification key: ' + key)
        value[key] = item
    return value


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_SIZE:
        raise QualificationError('Invalid or oversized qualification receipt')
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_JSON_SIZE:
            raise QualificationError('Qualification receipt exceeds JSON limit')
        return json.loads(raw, object_pairs_hook=_pairs), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, QualificationError):
            raise
        raise QualificationError('Invalid qualification receipt') from exc


def consumer_identity():
    """Identify actual imported consumer sources, including development checkouts."""
    from . import __version__
    root = Path(__file__).resolve().parent
    members = ('qualification.py', 'execution_fxb.py', 'graph_fxb.py',
               'checkpoint.py', 'native_fxb.py', 'settings.py', 'memory.py')
    if (root / 'native_lifetime.py').is_file():
        members += ('native_lifetime.py',)
    return {'name': 'fxbx', 'version': __version__,
            'source_sha256': _digest({name: _file_sha(root / name) for name in members}),
            'required_features': ['qualification_receipt_v1']}


def _checkpoint(weights, plan):
    from .checkpoint import SafeTensorWeights
    targets = {spec['target'] for spec in plan['tensors'].values()
               if spec['kind'] in ('parameter', 'buffer')}
    if not targets and not weights:
        return {'target_names': {}, 'tensor_files': {}, 'index_sha256': None}
    if not isinstance(weights, SafeTensorWeights):
        raise QualificationError('Qualified loading requires the recorded SafeTensorWeights checkpoint')
    if set(weights.names) != targets:
        raise QualificationError('Qualification checkpoint targets differ')
    digests = {}
    for file, identity in weights._identities.items():
        if weights._identity(file) != identity:
            raise QualificationError('Checkpoint changed before qualification admission')
        digests[file] = _file_sha(file)
        if weights._identity(file) != identity:
            raise QualificationError('Checkpoint changed during qualification admission')
    index = weights.path / 'model.safetensors.index.json' if weights.path.is_dir() else None
    return {'target_names': dict(weights.names),
            'tensor_files': {name: digests[file] for name, file in weights._files.items()},
            'index_sha256': _file_sha(index) if index is not None and index.exists() else None}


def qualification_bindings(fxb_path, plan, weights):
    """Return artifact/environment identity only; this does not qualify a model."""
    try:
        packages = {name: importlib.metadata.version(name)
                    for name in ('torch', 'furiosa-torch', 'furiosa-tcc')}
    except importlib.metadata.PackageNotFoundError as exc:
        raise QualificationError('Qualification SDK package unavailable: ' + str(exc)) from exc
    return {'fxb_sha256': _file_sha(fxb_path), 'execution_plan_sha256': _digest(plan),
            'consumer': consumer_identity(),
            'environment': {'packages': packages, 'compiler_identity': plan['compiler_identity'],
                            'device_config': plan['device_config']},
            'weights': _checkpoint(weights, plan)}


def validate_qualification(fxb_path, plan, weights, receipt_path):
    """Reject stale/wrong qualification before allocation; preserve raw loading."""
    receipt, digest = _read(receipt_path)
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA or receipt.get('status') != 'verified':
        raise QualificationError('Expected a verified producer qualification receipt v1')
    if receipt.get('route') != 'tcl_static' or plan.get('schema') != 'tclmachine.execution-plan.v1':
        raise QualificationError('Qualification route/plan schema mismatch')
    scope, evidence = receipt.get('scope'), receipt.get('evidence')
    if (not isinstance(scope, dict) or scope.get('kind') != 'fixed_shape_fixtures'
            or scope.get('entries') != ['forward'] or 'forward' not in plan['entries']
            or scope.get('output_devices') != ['cpu']
            or any(type(scope.get(n)) not in (int, float) or not math.isfinite(scope[n]) or scope[n] < 0 for n in ('atol', 'rtol'))
            or not isinstance(evidence, dict) or type(evidence.get('call_count')) is not int
            or evidence['call_count'] < 1 or type(evidence.get('cpu_routes')) is not int or evidence['cpu_routes'] != 0
            or evidence.get('fixture_readonly_checks') is not True):
        raise QualificationError('Incomplete qualification scope/evidence')
    current = qualification_bindings(fxb_path, plan, weights)
    if receipt.get('bindings') != current:
        raise QualificationError('Qualification does not match current FXB/plan/checkpoint/consumer/SDK')
    return {'status': 'verified', 'receipt_sha256': digest, 'scope': scope,
            'effects_scope': 'provided fixtures only; not universal alias or mutation proof'}
