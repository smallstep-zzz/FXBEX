"""Hash-bound generic EDF execution plans inside native FXBs.

Static graph contracts bind actual architecture, graph/EDF hashes and tensor ABI.
They use an explicit fxbX consumer, not Furiosa-LLM architecture dispatch.
"""
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
from typing import Protocol
import zipfile

from .native_fxb import (NativeFXBError, PRODUCER_MEMBER, _contract, _name,
                         build_native, validate_native)


class GraphFXBError(NativeFXBError):
    pass


def _bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise GraphFXBError(f'Graph contract must be finite JSON: {exc}') from exc


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


class NativeContractProvider(Protocol):
    def provide(self, *, model_metadata: Mapping, edf_export: Mapping) -> Mapping:
        """Return manifest, kernel_name, bucket, member, source, model_metadata_sha256.

        Providers are trusted host code responsible for obtaining the real model
        fingerprint and registering this EDF through a supported native producer.
        """


class OfficialNativeContractProvider:
    def provide(self, *, model_metadata, edf_export):
        raise GraphFXBError(
            'EXPORT_BRIDGE_GAP: the inspected official SDK exposes build_fxb(model_path, ...) '
            'but no public custom-graph manifest/fingerprint registration API. Supply an '
            'authentic NativeContractProvider; arbitrary model metadata cannot replace it.')


@dataclass
class ExistingNativeContract:
    """Explicit host-supplied native contract; never derived from a model name."""
    manifest: dict
    model_metadata: dict
    kernel_name: str
    bucket: dict
    member: str
    source: dict
    device_config: dict | None = None

    def provide(self, *, model_metadata, edf_export):
        if _sha(model_metadata) != _sha(self.model_metadata):
            raise GraphFXBError('Native contract belongs to different model metadata')
        return {'manifest': self.manifest, 'kernel_name': self.kernel_name,
                'bucket': self.bucket, 'member': self.member, 'source': self.source,
                'device_config': self.device_config,
                'model_metadata_sha256': _sha(self.model_metadata)}


def _spec(spec):
    if (not isinstance(spec, dict) or not isinstance(spec.get('shape'), list)
            or not isinstance(spec.get('stride'), list)
            or len(spec['shape']) != len(spec['stride']) or not isinstance(spec.get('dtype'), str)
            or any(type(x) is not int or x < 0 for x in spec['shape'] + spec['stride'])):
        raise GraphFXBError('Graph tensors require concrete shape, stride and dtype')


def _bindings(inputs, outputs, abi):
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not outputs:
        raise GraphFXBError('Graph requires ordered input/output bindings')
    names, user_indices = set(), []
    for binding in inputs:
        _spec(binding)
        name = binding.get('name', binding.get('fx_name'))
        if not isinstance(name, str) or not name or name in names:
            raise GraphFXBError('Input bindings require unique names')
        names.add(name)
        if binding.get('kind') == 'user_input':
            index = binding.get('user_index')
            if type(index) is not int or index < 0:
                raise GraphFXBError('User inputs require ordered integer user_index')
            user_indices.append(index)
        elif binding.get('kind') in ('parameter', 'buffer'):
            if not isinstance(binding.get('target'), str) or not binding['target']:
                raise GraphFXBError('External parameters/buffers require state_dict targets')
        else:
            raise GraphFXBError('Unsupported graph input kind; mutable state/constants require an explicit bridge')
    if sorted(user_indices) != list(range(len(user_indices))):
        raise GraphFXBError('User input indices must be unique and contiguous')
    for output in outputs:
        _spec(output)
    if (not isinstance(abi, dict) or not isinstance(abi.get('inputs'), list)
            or not isinstance(abi.get('outputs'), list)
            or len(abi['inputs']) != len(inputs) or len(abi['outputs']) != len(outputs)
            or any(not isinstance(x, str) or not x for x in abi['inputs'] + abi['outputs'])):
        raise GraphFXBError('Exact public EDF input/output ABI descriptions are required')


def _registration(contract, metadata):
    if not isinstance(contract, dict) or contract.get('model_metadata_sha256') != _sha(metadata):
        raise GraphFXBError('Native provider did not bind the target model metadata')
    manifest = contract.get('manifest')
    files, count = _contract(manifest)
    member = _name(contract.get('member'))
    name = contract.get('kernel_name')
    bucket = contract.get('bucket')
    if (files != {member} or count != 1 or not isinstance(name, str)
            or manifest['kernels'].get(name) != [{'bucket': bucket, 'file': member}]):
        raise GraphFXBError('Native contract must register this single graph EDF and exact bucket')
    if not isinstance(contract.get('source'), dict) or not contract['source']:
        raise GraphFXBError('Native provider provenance is required')
    architecture = metadata.get('architecture')
    if not architecture or manifest['model']['architecture'] != architecture:
        raise GraphFXBError('Real model architecture must match the native provider fingerprint')
    return manifest, {'name': name, 'bucket': bucket, 'file': member}


def export_graph(edf_export, output, *, model_metadata, contract_provider=None,
                 verify_official=True):
    """Assemble an EDF export with an automatic static or supplied native contract."""
    # Resolve before staging: an absent public native bridge must not emit a fake FXB.
    if contract_provider is None:
        from .static_graph_contract import StaticGraphContractProvider
        provider=StaticGraphContractProvider()
    else:provider=contract_provider
    contract = provider.provide(model_metadata=model_metadata, edf_export=edf_export)
    contract = json.loads(_bytes(contract))
    metadata = json.loads(_bytes(model_metadata))
    manifest, kernel = _registration(contract, metadata)
    inputs, outputs, abi = (edf_export.get(key) for key in ('inputs', 'outputs', 'abi'))
    _bindings(inputs, outputs, abi)
    source = Path(edf_export['edf_path'])
    if source.is_symlink() or not source.is_file() or source.suffix != '.edf':
        raise GraphFXBError('Expected a genuine compiled .edf file')
    digest = _file_sha(source)
    if digest != edf_export.get('edf_sha256'):
        raise GraphFXBError('Exported EDF checksum mismatch')
    plan = json.loads(_bytes({
        'schema': 'fxbex.graph-plan.v1', 'model_metadata': metadata,
        'model_metadata_sha256': _sha(metadata), 'native_manifest_sha256': _sha(manifest),
        'kernel': kernel, 'edf_sha256': digest, 'inputs': inputs, 'outputs': outputs,
        'abi': abi, 'compatibility': edf_export.get('compatibility', {}),
        'device_config': contract.get('device_config'),
        'expected_num_fusion': edf_export.get('expected_num_fusion'),
        'validation': edf_export.get('validation', {}),
        'target_npu': edf_export.get('target_npu'),
        'compiler_config': edf_export.get('compiler_config'),
        'output_tree_spec': edf_export.get('output_tree_spec'),
        'source_identity': edf_export.get('source_identity', {}),
        'graph_metadata': edf_export.get('graph_metadata', {}),
        'native_contract_source': contract['source']}))
    with tempfile.TemporaryDirectory(prefix='fxbex-graph-') as temporary:
        root = Path(temporary)
        (root / 'manifest.json').write_bytes(_bytes(manifest))
        destination = root / kernel['file']
        destination.parent.mkdir(parents=True)
        shutil.copyfile(source, destination)
        if _file_sha(destination) != digest:
            raise GraphFXBError('EDF changed during staging')
        report = build_native(root, output, verify_official=verify_official,
                              producer_metadata={'graph_plan': plan, 'graph_plan_sha256': _sha(plan)})
    report['graph_plan'] = plan
    report['gates']['graph_plan'] = 'passed'
    return report


def read_graph_plan(path, *, verify_official=False):
    """Validate references, ordered bindings and hashes without importing torch."""
    report = validate_native(path, verify_official=verify_official)
    producer = report['producer_metadata'] or {}
    plan = producer.get('graph_plan')
    if (not isinstance(plan, dict) or plan.get('schema') != 'fxbex.graph-plan.v1'
            or producer.get('graph_plan_sha256') != _sha(plan)):
        raise GraphFXBError('Missing or changed graph execution plan')
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if plan.get('native_manifest_sha256') != _sha(manifest):
            raise GraphFXBError('Graph plan native manifest changed')
        metadata = plan.get('model_metadata')
        if not isinstance(metadata, dict) or plan.get('model_metadata_sha256') != _sha(metadata):
            raise GraphFXBError('Graph plan target model metadata changed')
        _registration({'manifest': manifest, 'kernel_name': plan['kernel']['name'],
                       'member': plan['kernel']['file'], 'bucket': plan['kernel']['bucket'],
                       'model_metadata_sha256': _sha(metadata),
                       'source': plan.get('native_contract_source')}, metadata)
        _bindings(plan.get('inputs'), plan.get('outputs'), plan.get('abi'))
        digest = hashlib.sha256()
        with archive.open(plan['kernel']['file']) as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != plan.get('edf_sha256'):
            raise GraphFXBError('Graph plan EDF checksum mismatch')
    return plan


def _tensor_matches(value, spec, torch):
    return (isinstance(value, torch.Tensor) and list(value.shape) == spec['shape']
            and list(value.stride()) == spec['stride']
            and str(value.dtype).removeprefix('torch.') == spec['dtype'].removeprefix('torch.')
            and value.device.type in ('cpu', 'furiosa'))


class GraphFXB:
    """Explicit single-EDF consumer; not Furiosa-LLM architecture dispatch."""
    def __init__(self, path, *, weights=None, device='furiosa:0'):
        if device != 'furiosa:0':
            raise GraphFXBError('Only the explicit furiosa:0 device is supported')
        self.path, self.device = Path(path), device
        self.plan = read_graph_plan(path)
        self.weights = dict(weights or {})
        self.program = None
        self.failed = False
        self.stack = None
        self.resident_weights = {}

    def __enter__(self):
        if self.stack is not None:
            raise GraphFXBError('Graph is already loaded')
        self.stack = ExitStack()
        try:
            return self._load()
        except BaseException:
            self.close()
            raise

    def _load(self):
        import torch
        import furiosa.torch as sdk
        from furiosa.torch.config import set_cpu_fallback, get_cpu_fallback
        compatibility = self.plan['compatibility']
        packages = compatibility.get('packages', compatibility)
        for name in ('torch', 'furiosa-torch', 'furiosa-tcc'):
            expected = packages.get(name)
            if not isinstance(expected, str) or importlib.metadata.version(name) != expected:
                raise GraphFXBError(f'Graph SDK package mismatch: {name}')
        missing = {x['target'] for x in self.plan['inputs'] if x['kind'] != 'user_input'} - self.weights.keys()
        if missing:
            raise GraphFXBError(f'Missing external model weights/buffers: {sorted(missing)}')
        for binding in self.plan['inputs']:
            if binding['kind'] != 'user_input' and not _tensor_matches(self.weights[binding['target']], binding, torch):
                raise GraphFXBError(f'External weight/buffer metadata mismatch: {binding["target"]}')
        with zipfile.ZipFile(self.path) as archive:
            raw = archive.read(self.plan['kernel']['file'])
        if hashlib.sha256(raw).hexdigest() != self.plan['edf_sha256']:
            raise GraphFXBError('EDF changed before loading')
        config = self.plan['device_config']
        if config != {'fusion': 8, 'mesh': 'single'}:
            raise GraphFXBError('RUNTIME_BRIDGE_GAP: exact fusion8/single device configuration required')
        # EDF preprocessing uses this configuration, before device allocation.
        sdk.set_fusion(config['fusion'])
        sdk.set_mesh_kind(sdk.MeshKind.Single())
        runnable = sdk.ir.Runnable.from_serialized_edf(raw)
        actual = {'inputs': [str(x) for x in runnable.input_dram_shapes],
                  'outputs': [str(x) for x in runnable.output_dram_shapes]}
        if any(actual[key] != self.plan['abi'][key] for key in ('inputs', 'outputs')):
            raise GraphFXBError('Official EDF loader ABI differs from recorded export')
        expected_fusion = self.plan.get('expected_num_fusion')
        if (type(expected_fusion) is not int or expected_fusion != config['fusion']
                or getattr(runnable, 'expected_num_fusion', None) != expected_fusion):
            raise GraphFXBError('RUNTIME_BRIDGE_GAP: recorded/public EDF fusion configuration differs or is unknown')
        self.torch, self.sdk = torch, sdk
        # Verify a completion primitive before allocating device resources.
        if not callable(getattr(sdk, 'current_stream', None)):
            raise GraphFXBError('RUNTIME_BRIDGE_GAP: public stream completion API unavailable')
        from .settings import npu_lock
        self.stack.enter_context(npu_lock())
        set_cpu_fallback(False)
        if get_cpu_fallback():
            raise GraphFXBError('Cannot disable CPU model fallback')
        self.program = sdk.Runnable(runnable).to(self.device)
        self.resident_weights = {target: value.detach().to(self.device)
                                 for target, value in self.weights.items()
                                 if target in {x.get('target') for x in self.plan['inputs']}}
        self.failed = False
        return self

    def predict(self, inputs, *, output_device='cpu'):
        if self.program is None or self.failed:
            raise GraphFXBError('Graph must be loaded and healthy before execution')
        if output_device not in ('cpu', 'furiosa'):
            raise GraphFXBError('Output device must be cpu or furiosa')
        user = [x for x in self.plan['inputs'] if x['kind'] == 'user_input']
        if isinstance(inputs, Mapping):
            names = [x.get('name', x.get('fx_name')) for x in user]
            if set(inputs) != set(names):
                raise GraphFXBError('User input names differ from the exported graph')
            values = {x['user_index']: inputs[x.get('name', x.get('fx_name'))] for x in user}
        elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)) and len(inputs) == len(user):
            values = dict(enumerate(inputs))
        else:
            raise GraphFXBError('Expected exact ordered user tensors or named tensor mapping')
        arguments = []
        for binding in self.plan['inputs']:
            value = values[binding['user_index']] if binding['kind'] == 'user_input' else self.resident_weights[binding['target']]
            if not _tensor_matches(value, binding, self.torch) or value.requires_grad:
                raise GraphFXBError('Graph input/weight shape, stride, dtype or inference contract mismatch')
            arguments.append(value)
        try:
            with self.torch.no_grad():
                outputs = self.program(*(value.to(self.device) for value in arguments))
                if len(outputs) != len(self.plan['outputs']):
                    raise GraphFXBError('Graph output arity mismatch')
                if any(not _tensor_matches(value, spec, self.torch) for value, spec in zip(outputs, self.plan['outputs'])):
                    raise GraphFXBError('Graph output metadata mismatch')
                result = tuple(x.cpu() if output_device == 'cpu' else x for x in outputs)
                self._synchronize()
                return result
        except BaseException:
            self.failed = True
            raise

    def _synchronize(self):
        current = getattr(self.sdk, 'current_stream', None)
        stream = current(self.torch.device(self.device)) if callable(current) else None
        synchronize = getattr(stream, 'synchronize', None)
        if not callable(synchronize):
            raise GraphFXBError('RUNTIME_BRIDGE_GAP: public stream completion API unavailable')
        synchronize()

    def close(self):
        try:
            if self.program is not None:
                self._synchronize()
        finally:
            self.program = None
            self.resident_weights.clear()
            if self.stack is not None:
                self.stack.close()
                self.stack = None

    def __exit__(self, *args):
        self.close()
