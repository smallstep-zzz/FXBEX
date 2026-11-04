"""Exact-request replay of SDK Runnable bytes, not standalone native EDF execution."""
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re


class ReplayError(RuntimeError):
    pass


@dataclass
class ReplayStats:
    compile_calls: int = 0
    replay_hits: int = 0


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _compatibility(torch, sdk):
    versions = {}
    for name in ('furiosa-torch', 'furiosa-tcc'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {}
    if getattr(sdk, '__file__', None):
        directory = Path(sdk.__file__).parent
        for relative in ('compiler/__init__.py', '_module/runnable.py', '_backend/torch_compile.py'):
            path = directory / relative
            sources[relative] = _hash(path.read_bytes()) if path.is_file() else None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'sdk': versions, 'sdk_sources': sources}


def _compatible_runtime(recorded, current):
    """Allow Python minor-version changes, while pinning the SDK/torch ABI."""
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    recorded_python = recorded.get('python', '')
    current_python = current.get('python', '')
    if str(recorded_python).split('.', 1)[0] != str(current_python).split('.', 1)[0]:
        return False
    return ({key: value for key, value in recorded.items() if key != 'python'} ==
            {key: value for key, value in current.items() if key != 'python'})


def _value(value, torch, *, tensor_data=False):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return {'float': value.hex()}
    if isinstance(value, torch.fx.Node):
        return {'node': value.name}
    if isinstance(value, torch.Tensor):
        result = {'tensor': {'shape': [_value(x, torch) for x in value.shape],
                            'stride': [_value(x, torch) for x in value.stride()],
                            'dtype': str(value.dtype), 'device': str(value.device),
                            'layout': str(value.layout), 'requires_grad': value.requires_grad,
                            'storage_offset': _value(value.storage_offset(), torch),
                            'conjugate': value.is_conj(), 'negative': value.is_neg()}}
        if tensor_data:
            if value.numel() * value.element_size() > 1024 * 1024:
                raise ReplayError('Constant tensor exceeds 1 MiB; lift it into an input')
            if value.device.type != 'cpu' or value.layout != torch.strided:
                raise ReplayError('Only dense CPU tensor constants are supported')
            try:
                raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            except Exception as exc:
                raise ReplayError('Cannot read tensor constant') from exc
            result['sha256'] = _hash(raw)
        return result
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return {'torch_type': str(value)}
    if isinstance(value, Enum):
        return {'enum': type(value).__module__ + '.' + type(value).__qualname__, 'name': value.name}
    if is_dataclass(value) and not isinstance(value, type):
        return {'dataclass': type(value).__module__ + '.' + type(value).__qualname__,
                'fields': {f.name: _value(getattr(value, f.name), torch, tensor_data=tensor_data)
                           for f in fields(value)}}
    if type(value) in (list, tuple, torch.fx.immutable_collections.immutable_list):
        return {type(value).__name__: [_value(x, torch, tensor_data=tensor_data) for x in value]}
    if type(value) in (dict, torch.fx.immutable_collections.immutable_dict) and all(type(k) is str for k in value):
        return {k: _value(v, torch, tensor_data=tensor_data) for k, v in value.items()}
    if isinstance(value, slice):
        return {'slice': [_value(v, torch) for v in (value.start, value.stop, value.step)]}
    raise ReplayError(f'Unsupported key value: {type(value).__module__}.{type(value).__qualname__}')


def _request(ep, config, task_name, torch):
    if config is not None:
        if not type(config).__module__.startswith('furiosa.'):
            raise ReplayError('Unsupported compiler configuration type')
        config_text = str(config)
        if re.search(r' at 0x[0-9a-fA-F]+', config_text):
            raise ReplayError('Compiler configuration has no stable string form')
        config = {'type': type(config).__module__ + '.' + type(config).__qualname__, 'value': config_text}
    nodes = []
    for node in ep.graph.nodes:
        target = node.target
        if not isinstance(target, str):
            if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
                target = str(target)
            elif getattr(target, '__module__', None) in ('operator', '_operator', 'builtins'):
                target = target.__module__ + '.' + target.__qualname__
            else:
                raise ReplayError(f'Unsupported graph target: {target!r}')
        if node.op in ('get_attr', 'call_module'):
            raise ReplayError('Graph attributes/modules must be lifted into explicit inputs')
        nodes.append({'name': node.name, 'op': node.op, 'target': target,
                      'args': _value(node.args, torch, tensor_data=True),
                      'kwargs': _value(node.kwargs, torch, tensor_data=True),
                      'value': _value(node.meta.get('val'), torch)})
    return {'nodes': nodes, 'signature': _value(ep.graph_signature, torch),
            'constants': _value(ep.constants, torch, tensor_data=True),
            'config': config}


@contextmanager
def _session(root, recording, guard_native=True):
    torch = importlib.import_module('torch')
    compiler = importlib.import_module('furiosa.torch.compiler')
    sdk = importlib.import_module('furiosa.torch')
    ir = sdk.ir
    root = Path(root)
    plan_path = root / 'execution_plan.json'
    compatibility = _compatibility(torch, sdk)
    if recording:
        if plan_path.exists():
            raise ReplayError('Refusing to overwrite an existing execution plan')
        (root / 'edfs').mkdir(parents=True, exist_ok=True)
        plan = {'format': 'sdk-runnable', 'version': 1, 'compatibility': compatibility, 'steps': []}
    else:
        plan = json.loads(plan_path.read_text())
        if (plan.get('format') != 'sdk-runnable' or plan.get('version') != 1
                or not _compatible_runtime(plan.get('compatibility'), compatibility)):
            raise ReplayError('Runnable plan format or environment compatibility mismatch')
    records = {}
    for record in plan['steps']:
        key = record['key']
        if key in records or _hash(_json(record['args']).encode()) != key:
            raise ReplayError('Duplicate or invalid request key')
        if record['path'] != f'edfs/{key}.runnable':
            raise ReplayError('Invalid runnable path')
        records[key] = record
    original = compiler.compile
    stats = ReplayStats()

    def compile(ep, compiler_config=None, *, task_name=None):
        effective_config = compiler_config if compiler_config is not None else sdk.config.get_compiler_config()
        args = _request(ep, effective_config, task_name, torch)
        key = _hash(_json(args).encode())
        if recording:
            stats.compile_calls += 1
            runnable = original(ep, compiler_config, task_name=task_name)
            data = bytes(runnable.serialize())
            record = {'key': key, 'path': f'edfs/{key}.runnable', 'sha256': _hash(data), 'args': args}
            if key in records and records[key] != record:
                raise ReplayError('Identical request produced different runnable bytes')
            if key not in records:
                path = root / record['path']
                if path.exists() and path.read_bytes() != data:
                    raise ReplayError('Conflicting existing runnable file')
                path.write_bytes(data)
                records[key] = record
                plan['steps'].append(record)
            return runnable
        if key not in records:
            raise ReplayError(f'No recorded runnable for request {key}; compilation is forbidden')
        record = records[key]
        path = root / record['path']
        if path.resolve().parent != (root / 'edfs').resolve() or path.is_symlink():
            raise ReplayError('Runnable path escapes package')
        data = path.read_bytes()
        if _hash(data) != record['sha256']:
            raise ReplayError('Runnable checksum mismatch')
        runnable = ir.Runnable.deserialize(data)
        stats.replay_hits += 1
        return runnable

    def forbidden(*args, **kwargs):
        raise ReplayError('Native compiler entry point is forbidden during replay')

    patches = [(compiler, 'compile', original)]
    try:
        compiler.compile = compile
        if not recording and guard_native:
            native = importlib.import_module('furiosa.torch._C')
            for name in ('compile', 'compile_raw_dfg'):
                if hasattr(native.compiler, name):
                    patches.append((native.compiler, name, getattr(native.compiler, name)))
                    setattr(native.compiler, name, forbidden)
        yield stats
        if recording:
            plan_path.write_text(_json(plan) + '\n')
    finally:
        for obj, name, value in reversed(patches):
            setattr(obj, name, value)


def capture(root):
    """Capture public compile calls and write an execution plan on successful exit."""
    return _session(root, True)


def replay(root, *, guard_native=True):
    """Restore exact captured requests; tracing/lowering still occurs, compilation cannot."""
    return _session(root, False, guard_native)
