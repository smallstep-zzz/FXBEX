"""Logical memory plans from small metadata, never checkpoint/binary inspection.

These are logical estimates: SDK padding, aliases, executable workspaces and allocator
reservations are deliberately not presented as measured physical HBM usage.
"""
import hashlib
import json
import math
from pathlib import Path

from .package import PackageError, DEFAULT_MAX_JSON_SIZE


_SIZES = {'bool': 1, 'uint8': 1, 'int8': 1, 'float8_e4m3fn': 1,
          'float8_e5m2': 1, 'int16': 2, 'float16': 2, 'bfloat16': 2,
          'int32': 4, 'float32': 4, 'int64': 8, 'float64': 8}


def tensor_bytes(spec):
    """Unknown/dynamic shape or dtype stays unknown; no guessed padding/layout."""
    if not isinstance(spec, dict):
        return None
    shape, dtype = spec.get('shape'), spec.get('dtype')
    if not isinstance(shape, list) or len(shape) > 64:
        return None
    if any(type(n) is not int or n < 0 for n in shape):
        return None
    size = _SIZES.get(str(dtype).removeprefix('torch.'))
    return math.prod(shape) * size if size is not None else None


def _policy(value):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise PackageError('memory policy must be an object')
    result = dict(value)
    for name, default in [('release_threshold_bytes', 64 * 1024**2),
                          ('reserve_bytes', 0), ('safety_margin_bytes', 0),
                          ('budget_bytes', None)]:
        item = result.setdefault(name, default)
        if item is None and name == 'budget_bytes':
            continue
        if type(item) is not int or item < 0:
            raise PackageError(f'memory policy {name} must be a nonnegative integer')
    return result


def check_budget(known_bytes, policy):
    """Enforce an explicit logical budget; it is not a physical OOM diagnosis."""
    policy = _policy(policy)
    required = known_bytes + policy['reserve_bytes'] + policy['safety_margin_bytes']
    budget = policy['budget_bytes']
    if budget is not None and required > budget:
        raise PackageError(f'Logical memory estimate {required} bytes exceeds configured budget {budget} bytes; '
                           'reduce shape/context/concurrency or change the execution plan')
    return {'known_required_bytes': required, 'budget_bytes': budget,
            'verdict': 'not_configured' if budget is None else 'logical_estimate_within_budget_physical_fit_unknown'}


def dag_plan(execution, config, index, previous=None):
    """Find last consumers and conservative out-of-place live tensor peaks."""
    if execution.get('format') != 'runnable-dag-v1':
        raise PackageError('Memory liveness requires runnable-dag-v1')
    previous = previous or {}
    policy = _policy(previous.get('policy'))
    steps = execution['steps']
    specs = dict(config.get('tensor_specs', {}))
    specs.update(config['inputs'])
    constants, undeclared_weights = set(), False
    for weight in index['weights']:
        tensors = weight.get('tensors')
        if tensors is None:
            undeclared_weights = True
            continue
        if not isinstance(tensors, (list, dict)):
            raise PackageError('Weight tensor bindings must be a list or object')
        for name in tensors:
            if name in constants or name in config['inputs']:
                raise PackageError('Duplicate memory tensor binding')
            constants.add(name)
            if isinstance(tensors, dict):
                specs[name] = tensors[name]
    bindings = {'execution': execution, 'inputs': config['inputs'], 'outputs': config['outputs'],
                'tensor_specs': config.get('tensor_specs', {}),
                'weights': [w.get('tensors') for w in index['weights']]}
    signature = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    if previous.get('kind') == 'dag-logical-memory-v1' and previous.get('binding_sha256') != signature:
        raise PackageError('Stale memory plan: execution/tensor metadata changed; regenerate the plan')
    last = {name: -1 for name in set(config['inputs']) | constants}
    ids = set()
    for i, step in enumerate(steps):
        if not isinstance(step.get('id'), str) or step['id'] in ids:
            raise PackageError('Memory plan requires unique step ids')
        ids.add(step['id'])
        for name in step['inputs']:
            if name not in last:
                if not undeclared_weights:
                    raise PackageError(f'Unbound or forward DAG input: {name}')
                constants.add(name)
            last[name] = i
        for name in step['outputs']:
            if name in last:
                raise PackageError(f'Duplicate DAG tensor binding: {name}')
            last[name] = i
    for name in config['outputs']:
        if name not in last:
            raise PackageError(f'Unbound DAG output: {name}')
        last[name] = len(steps)
    sizes = {name: tensor_bytes(specs.get(name)) for name in last}
    release = [[] for _ in steps]
    for name, consumer in last.items():
        if name not in constants and consumer < len(steps) and steps:
            release[max(consumer, 0)].append(name)
    live = set(config['inputs'])
    peak = sum(sizes[n] or 0 for n in live)
    retained = peak
    for step, dead in zip(steps, release):
        live.update(step['outputs'])
        peak = max(peak, sum(sizes[n] or 0 for n in live))
        retained += sum(sizes[n] or 0 for n in step['outputs'])
        live.difference_update(dead)
    live = set(config['inputs'])
    pending = set()
    batched_peak = sum(sizes[n] or 0 for n in live)
    for step, dead in zip(steps, release):
        live.update(step['outputs'])
        batched_peak = max(batched_peak, sum(sizes[n] or 0 for n in live))
        pending.update(dead)
        if sum(sizes[n] or 0 for n in pending) >= policy['release_threshold_bytes'] or len(pending) >= 64:
            live.difference_update(pending)
            pending.clear()
    weights = sum(sizes[n] or 0 for n in constants)
    unknown = sorted(n for n, size in sizes.items() if size is None)
    estimate = {'weight_logical_bytes': weights, 'peak_live_tensor_logical_bytes': peak,
                'planned_batched_peak_tensor_logical_bytes': batched_peak,
                'retained_until_return_tensor_logical_bytes': retained,
                'known_logical_peak_bytes': weights + batched_peak, 'unknown_tensors': unknown,
                'undeclared_weight_tensors': undeclared_weights,
                'physical_allocated_bytes': None,
                'note': 'Out-of-place tensor-size estimate; aliases may be double counted. '
                        'Excludes physical padding, replicas, kernel workspace and allocator reserves. '
                        'Batched completion can keep dead tensors alive beyond their last consumer.'}
    return {**previous, 'schema_version': 1, 'kind': 'dag-logical-memory-v1',
            'allocator': 'SDK', 'binding_sha256': signature, 'policy': policy,
            'last_use': last, 'release_after_step': [sorted(names) for names in release],
            'input_names': list(config['inputs']), 'produced_after_step': [s['outputs'] for s in steps],
            'estimate': estimate, 'admission': check_budget(weights + batched_peak, policy)}


def stateful_plan(weight_logical_bytes, states, *, policy=None):
    """Adapter-supplied real state shapes, including heterogeneous recurrent/KV layers."""
    if type(weight_logical_bytes) is not int or weight_logical_bytes < 0:
        raise PackageError('weight_logical_bytes must be a nonnegative integer')
    current = maximum = 0
    unknown = []
    for state in states:
        size = tensor_bytes(state)
        resident, limit = state['resident_copies'], state['maximum_copies']
        if any(type(n) is not int or n < 0 for n in (resident, limit)) or resident > limit:
            raise PackageError('Invalid resident/maximum state copies')
        if size is None:
            unknown.append(state['name'])
        else:
            current += size * resident
            maximum += size * limit
    policy = _policy(policy)
    return {'schema_version': 1, 'kind': 'adapter-logical-memory-v1', 'allocator': 'SDK',
            'weight_logical_bytes': weight_logical_bytes, 'states': states, 'policy': policy,
            'estimate': {'persistent_current_logical_bytes': current,
                         'persistent_maximum_logical_bytes': maximum,
                         'known_logical_peak_bytes': weight_logical_bytes + maximum,
                         'unknown_states': unknown, 'physical_allocated_bytes': None,
                         'note': 'Weights plus declared state only; execution workspace, physical layout, '
                                 'padding and allocator reserves are unknown. Lazy pages are not preallocated.'},
            'admission': check_budget(weight_logical_bytes + maximum, policy)}


def read_inputs(root):
    result = []
    for name in ('execution_plan.json', 'runtime_config.json', 'weights_index.json', 'memory_plan.json'):
        path = Path(root) / name
        if path.is_symlink() or path.stat().st_size > DEFAULT_MAX_JSON_SIZE:
            raise PackageError(f'Invalid or oversized memory metadata: {name}')
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise PackageError(f'{name} must be an object')
        result.append(value)
    return result


def analyze(root):
    execution, config, index, memory = read_inputs(root)
    return analyze_documents(execution, config, index, memory)


def analyze_documents(execution, config, index, memory):
    if execution.get('format') == 'runnable-dag-v1':
        return dag_plan(execution, config, index, memory)
    if memory.get('kind') == 'adapter-logical-memory-v1':
        return {**memory, **stateful_plan(memory['weight_logical_bytes'], memory['states'],
                                        policy=memory.get('policy'))}
    return {'schema_version': 1, 'kind': 'unplanned', 'physical_allocated_bytes': None,
            'reason': 'Adapter must declare actual weight/state/workspace shapes; no model-name heuristics',
            'legacy_metadata': memory}


def inspect_archive(path):
    """Read/hash only small metadata, so planning never scans multi-GB weights."""
    import zipfile
    names = ('execution_plan.json', 'runtime_config.json', 'weights_index.json', 'memory_plan.json')
    with zipfile.ZipFile(path) as archive:
        def read(name):
            if archive.namelist().count(name) != 1 or archive.getinfo(name).file_size > DEFAULT_MAX_JSON_SIZE:
                raise PackageError(f'Duplicate or oversized memory metadata: {name}')
            return archive.read(name)
        manifest = json.loads(read('manifest.json'))
        documents = []
        for name in names:
            data = read(name)
            if hashlib.sha256(data).hexdigest() != manifest['files'][name]['sha256']:
                raise PackageError(f'Memory metadata checksum mismatch: {name}')
            documents.append(json.loads(data))
    return {**analyze_documents(*documents),
            'integrity_scope': 'memory metadata only; model artifact integrity not checked by this analysis'}


def summarize(plan):
    """Bounded terminal/agent output; detailed liveness stays in the JSON file."""
    result = {key: plan[key] for key in ('schema_version', 'kind', 'admission', 'reason') if key in plan}
    estimate = dict(plan.get('estimate', {}))
    for key in ('unknown_tensors', 'unknown_states'):
        if key in estimate:
            estimate[key + '_count'] = len(estimate[key])
            estimate[key] = estimate[key][:5]
    result['estimate'] = estimate
    result['release_steps'] = len(plan.get('release_after_step', []))
    result['next_action'] = ('declare_adapter_shapes' if plan['kind'] == 'unplanned' else
                             'validate_shapes_numerics_trace_and_repeated_lifetimes_on_NPU')
    return result


class TensorLifetime:
    """Drop host references only after completion of their stream consumers.

    A bounded batch amortizes host synchronization. No custom arena, aliases or
    allocator-level buffer reuse is introduced; the SDK remains the allocator.
    """
    def __init__(self, plan, constants, synchronize=None):
        self.plan, self.constants, self.synchronize = plan, set(constants), synchronize
        self.pending = set()
        self.peak = self.released = self.barriers = 0
        self.sizes = None
        self.live_bytes = self.pending_bytes = 0

    def after_step(self, index, values):
        if self.sizes is None:
            self.sizes = {n: values[n].numel() * values[n].element_size() for n in self.plan['input_names']}
            self.live_bytes = sum(self.sizes.values())
        for name in self.plan['produced_after_step'][index]:
            self.sizes[name] = values[name].numel() * values[name].element_size()
            self.live_bytes += self.sizes[name]
        self.peak = max(self.peak, self.live_bytes)
        for name in self.plan['release_after_step'][index]:
            if name in values and name not in self.pending:
                self.pending.add(name)
                self.pending_bytes += self.sizes[name]
        # Unknown completion semantics must retain the legacy lifetime.
        if self.synchronize is None:
            return
        if self.pending and (self.pending_bytes >= self.plan['policy']['release_threshold_bytes']
                             or len(self.pending) >= 64):
            self.synchronize()
            self.barriers += 1
            for name in self.pending:
                del values[name]
                self.live_bytes -= self.sizes.pop(name)
                self.released += 1
            self.pending.clear()
            self.pending_bytes = 0

    def summary(self):
        return {'peak_live_tensor_logical_bytes': self.peak, 'released_tensor_bindings': self.released,
                'completion_barriers': self.barriers, 'physical_allocated_bytes': None,
                'release_policy': 'stream_completed_batches' if self.synchronize else 'legacy_retention_no_completion_api',
                'note': 'Logical tensor references only; storage aliases may be double counted. '
                        'Dropping references does not prove driver memory was returned.'}
