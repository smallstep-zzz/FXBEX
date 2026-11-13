"""Fixed-plan runtime. Only built-in, installed adapters may execute a package."""
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings
import time
from .package import extract, PackageError
from .settings import lock_path, runs_path


def environment():
    versions = {}
    for name in ('furiosa-torch', 'furiosa-tcc', 'torch', 'transformers'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return versions


def doctor():
    try:
        p = subprocess.run(['furiosa-smi', 'status'], capture_output=True, text=True, timeout=15)
        device = {'returncode': p.returncode, 'status': p.stdout.strip(), 'error': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        device = {'error': str(exc)}
    return {'packages': environment(), 'device': device, 'format': 'FXB_EX',
            'note': 'Detection is not NPU execution acceptance; packages require matching SDK.'}


def resolve_adapter(name):
    if name == 'runnable_dag':return RunnableDAG
    entries=[ep for ep in importlib.metadata.entry_points(group='fxbex.adapters') if ep.name==name]
    if len(entries)!=1:
        raise PackageError(f'Adapter {name!r} requires exactly one installed fxbex.adapters plugin (found {len(entries)}). Install the matching model provider.')
    return entries[0].load()


def package_compatibility(manifest, *, allow_sdk_mismatch=False):
    actual = environment()
    notices = []
    required = manifest['compatibility'].get('packages', {})
    missing = [f'{name}=={expected}' for name, expected in required.items() if actual.get(name) is None]
    if missing:
        raise PackageError(
            f'Missing required packages in Python {sys.executable}: {", ".join(missing)}. '
            'Use a compatible SDK Python with fxbX installed '
            '(set FXBEX_WORKER_PYTHON to its executable).'
        )
    if (manifest.get('model',{}).get('name')=='RedHatAI/Qwen3.5-9B-FP8-dynamic'
            and sys.version_info[:2]==(3,11) and actual.get('furiosa-torch')=='2026.4.0'
            and actual.get('torch')=='2.10.0+cpu' and actual.get('transformers')=='5.22.1'):
        raise PackageError('Known unstable Qwen runtime combination: Python 3.11, SDK 2026.4.0, transformers 5.22.1 caused native crashes. Use the validated transformers==5.20.2 in this environment.')
    for name, expected in required.items():
        installed=actual.get(name)
        if installed == expected:continue
        if (name=='transformers' and isinstance(installed,str) and isinstance(expected,str)
                and installed.split('.')[0].isdigit() and installed.split('.')[0]==expected.split('.')[0]):
            notices.append(f'transformers differs: package={expected}, installed={installed}; proceeding with adapter validation; compatibility is not yet established')
        elif allow_sdk_mismatch:
            notices.append(
                f'SDK mismatch {name}: package={expected}, installed={installed}; '
                'attempting execution with this unverified combination'
            )
        else:
            raise PackageError(
                f'SDK mismatch {name} in Python {sys.executable}: package={expected}, installed={installed}. '
                'Use the matching SDK Python or pass --allow-sdk-mismatch to attempt execution.'
            )
    return notices


def check_compatibility(manifest, *, allow_sdk_mismatch=False, report_warning=None):
    for message in package_compatibility(manifest, allow_sdk_mismatch=allow_sdk_mismatch):
        if report_warning is None:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            report_warning(message)
    if manifest['compatibility'].get('hardware', 'rngd') != 'rngd' or manifest['compatibility'].get('fusion',8) != 8:
        raise PackageError('This adapter version requires RNGD fusion8')
    from .bundled import specification
    if specification(manifest) is None:resolve_adapter(manifest['runtime']['adapter'])


class Session:
    def __init__(self, path, *, report_dir=None, weights_dir=None,
                 allow_sdk_mismatch=False, report_warning=None):
        self.allow_sdk_mismatch = allow_sdk_mismatch
        self.report_warning = report_warning
        self.path = Path(path)
        self.weights_dir = Path(weights_dir).resolve() if weights_dir else None
        self.stack = ExitStack()
        self.report_dir = Path(report_dir or runs_path()/f'fxbex_{time.time_ns()}').resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = None
        self.telemetry = None
        self.memory = None
        self.failed = False

    def __enter__(self):
        try:
            self.root = self.stack.enter_context(extract(self.path))
            self.manifest = json.loads((self.root/'manifest.json').read_text())
            check_compatibility(self.manifest, allow_sdk_mismatch=self.allow_sdk_mismatch,
                                report_warning=self.report_warning)
            from .validation import preflight
            preflight(self.root, self.manifest, self.weights_dir)
            from .memory import analyze
            memory_plan = analyze(self.root)
            (self.report_dir/'memory_plan.json').write_text(json.dumps(memory_plan, indent=2))
            from .execution import ExecutionRecorder, MemoryTelemetry
            self.execution = ExecutionRecorder(self.report_dir, self.manifest)
            self.telemetry = MemoryTelemetry(self.report_dir)
            self.memory = self.telemetry.sample('before_load')
            # Share the canonical lock with web workers. Never assume PE partitioning.
            import fcntl
            from .settings import npu_lock
            try: self.stack.enter_context(npu_lock())
            except BlockingIOError: raise RuntimeError('NPU busy: unload the web model before loading FXB_EX')
            kind = self.manifest['runtime']['adapter']
            from .bundled import activate
            adapter=self.stack.enter_context(activate(self.root,self.manifest)) or resolve_adapter(kind)
            if kind=='runnable_dag':
                self.adapter=adapter(self.root,weights_dir=self.weights_dir)
            else:
                self.adapter=adapter(self.root,self.report_dir,self.stack,weights_dir=self.weights_dir)
            self.sample_memory('after_load')
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *args):
        try:
            if isinstance(self.adapter, RunnableDAG):
                self.adapter.close()
        finally:
            self.adapter = None
            # Release graph caches before yielding the canonical device lock.
            import sys, gc
            torch = sys.modules.get('torch')
            try:
                if torch is not None:
                    torch._dynamo.reset()
                gc.collect()
            finally:
                self.stack.close()
                if self.telemetry is not None:
                    self.memory = self.telemetry.sample('after_unload')

    def sample_memory(self, phase):
        if self.telemetry is not None:
            self.memory = self.telemetry.sample(phase)
            summary=getattr(self.adapter,'memory_summary',None)
            if callable(summary):
                try:
                    self.memory['model_logical_memory']=summary()
                    (self.report_dir/'memory_latest.json').write_text(json.dumps(self.memory,indent=2))
                except Exception as exc:self.memory['model_memory_error']=type(exc).__name__
        return self.memory

    def _call(self, operation, callback):
        if self.failed:
            raise RuntimeError('Session failed during execution; unload and reload before reuse')
        self.sample_memory(f'before_{operation}')
        try:
            return self.execution.call(operation, callback)
        except BaseException:
            # Arbitrary adapter exceptions can happen after state mutation; their
            # Python type alone cannot prove a harmless pre-dispatch input error.
            self.failed = True
            raise
        finally:
            self.sample_memory(f'after_{operation}')

    def predict(self, inputs):
        if not hasattr(self.adapter, 'predict'):raise ValueError('Package does not provide predict')
        return self._call('predict', lambda: self.adapter.predict(inputs))

    def predict_tensors(self, inputs, *, output_device='cpu'):
        if not hasattr(self.adapter,'predict_tensors'):raise NotImplementedError('Adapter has no tensor execution contract')
        validate=getattr(self.adapter,'validate_tensor_inputs',None)
        if callable(validate):validate(inputs)
        return self._call('predict_tensors',lambda:self.adapter.predict_tensors(inputs,output_device=output_device))

    def chat(self, messages, max_new_tokens=32):
        if not hasattr(self.adapter, 'chat'): raise ValueError('Package does not provide chat')
        return self._call('chat', lambda: self.adapter.chat(messages, max_new_tokens))

    def generate(self, prompt, max_tokens, on_token=None):
        if self.manifest['runtime']['adapter'] != 'qwen35_replay':
            raise NotImplementedError('This adapter does not implement incremental generation')
        if type(max_tokens) is not int or max_tokens < 0 or not isinstance(prompt,(str,list)):
            raise ValueError('generate requires text/messages and a nonnegative integer max_tokens')
        from .hug_engine import generate
        return self._call('generate', lambda: generate(
            self.adapter, prompt, max_tokens, on_token=on_token))


class RunnableDAG:
    """Typed ordered DAG with input/weight/state/output bindings; no model CPU ops."""
    def __init__(self, root, weights_dir=None):
        from .memory import analyze
        self.memory_plan = analyze(root)  # Validate/budget before any NPU upload.
        self.last_memory = None
        import torch
        from furiosa.torch import ir, Runnable
        from furiosa.torch import set_fusion,set_mesh_kind,MeshKind
        from furiosa.torch.config import set_cpu_fallback,get_cpu_fallback
        set_cpu_fallback(False)
        if get_cpu_fallback():raise RuntimeError('Cannot disable CPU fallback')
        set_fusion(8);set_mesh_kind(MeshKind.Single())
        self.torch = torch
        import furiosa.torch as sdk
        self.current_stream = getattr(sdk, 'current_stream', None)
        self.config = json.loads((root/'runtime_config.json').read_text())
        plan = json.loads((root/'execution_plan.json').read_text())
        if plan.get('format') != 'runnable-dag-v1': raise PackageError('Expected runnable-dag-v1 plan')
        device_config = plan.get('device_config', {'fusion': 8, 'mesh': 'single'})
        if (not isinstance(device_config, dict) or type(device_config.get('fusion')) is not int
                or device_config != {'fusion': 8, 'mesh': 'single'}):
            raise PackageError('Runnable DAG requires device_config fusion=8, mesh=single')
        self.steps = plan['steps']
        programs = {}
        for step in self.steps:
            name = step['id']
            artifact = (root/step['artifact']).resolve()
            if not artifact.is_relative_to(root.resolve()):
                raise PackageError('Artifact escapes package')
            program = ir.Runnable.deserialize(artifact.read_bytes())
            fusion = getattr(program, 'expected_num_fusion', None)
            if type(fusion) is not int or fusion != device_config['fusion']:
                raise PackageError(f'Runnable step {name} fusion mismatch: compiled={fusion}, '
                                   f'runtime={device_config["fusion"]}; recompile with matching fusion')
            programs[name] = program
        self.constants = {}
        index = json.loads((root/'weights_index.json').read_text())
        for item in index['weights']:
            path = root/item['path'] if 'path' in item else (Path(weights_dir)/item.get('name',Path(item['external_path']).name) if weights_dir else Path(item['external_path']))
            from .package import CHUNK
            import hashlib
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(CHUNK), b''): h.update(block)
            expected=item.get('sha256')
            if expected is None and 'path' in item:
                expected=json.loads((root/'manifest.json').read_text())['files'][item['path']]['sha256']
            if h.hexdigest() != expected: raise PackageError('External/internal weight checksum mismatch')
            tensors=torch.load(path,map_location='cpu',weights_only=True)
            for name in tensors: self.constants[name]=tensors[name].to('furiosa:0')
            del tensors
        self.weight_logical_bytes = sum(t.numel()*t.element_size() for t in self.constants.values())
        known=set(self.config['inputs']) | set(self.constants)
        self.modules={}
        for step in self.steps:
            if any(name not in known for name in step['inputs']): raise PackageError('Unbound or forward DAG input')
            if any(name in known for name in step['outputs']): raise PackageError('Duplicate DAG tensor binding')
            self.modules[step['id']] = Runnable(programs.pop(step['id'])).to('furiosa:0')
            known.update(step['outputs'])
        if any(name not in known for name in self.config['outputs']): raise PackageError('Unbound DAG output')

    def predict(self, inputs):
        if set(inputs)!=set(self.config['inputs']):raise ValueError('Input names do not match package')
        tensors={name:self.torch.tensor(inputs[name],dtype=getattr(self.torch,spec['dtype'])) for name,spec in self.config['inputs'].items()}
        result=self.predict_tensors(tensors,output_device='cpu',_validate_outputs=False)
        result['outputs']={name:value.tolist() for name,value in result['outputs'].items()}
        return result

    def validate_tensor_inputs(self, inputs):
        torch=self.torch
        if set(inputs)!=set(self.config['inputs']):raise ValueError('Input names do not match package')
        for name,spec in self.config['inputs'].items():
            tensor=inputs[name]
            if not isinstance(tensor,torch.Tensor):raise TypeError('Tensor API requires Tensor inputs')
            if list(tensor.shape)!=spec['shape'] or tensor.dtype!=getattr(torch,spec['dtype']):raise ValueError(f'Input shape/dtype mismatch: {name}')
            if spec.get('stride') is not None and list(tensor.stride())!=spec['stride']:raise ValueError(f'Input stride mismatch: {name}')
            if tensor.requires_grad:raise ValueError('Compiled execution is inference-only')
            if tensor.device.type not in ('cpu','furiosa'):raise ValueError('Unsupported input device')

    def predict_tensors(self, inputs, *, output_device="cpu", _validate_outputs=True):
        from .memory import TensorLifetime
        torch=self.torch
        self.validate_tensor_inputs(inputs)
        if output_device not in ('cpu','furiosa'):raise ValueError('Output device must be cpu or furiosa')
        values=dict(self.constants)
        for name, spec in self.config['inputs'].items():
            tensor=inputs[name]
            values[name]=tensor if tensor.device.type=='furiosa' else tensor.to('furiosa:0')
        if self.config['inputs']:
            del tensor
        stream = self.current_stream(torch.device('furiosa:0')) if callable(self.current_stream) else None
        synchronize = getattr(stream, 'synchronize', None)
        lifetime = TensorLifetime(self.memory_plan, self.constants, synchronize if callable(synchronize) else None)
        started=time.perf_counter()
        with torch.no_grad():
            for index, step in enumerate(self.steps):
                outputs=self.modules[step['id']](*(values[n] for n in step['inputs']))
                if len(outputs)!=len(step['outputs']): raise PackageError('Kernel output arity mismatch')
                values.update(zip(step['outputs'],outputs))
                del outputs
                lifetime.after_step(index, values)
            if output_device=='furiosa':
                if not callable(synchronize):raise RuntimeError('Resident output requires public stream completion API')
                synchronize()
            if _validate_outputs:
                for name in self.config['outputs']:
                    spec=self.config.get('tensor_specs',{}).get(name)
                    if spec is None:raise NotImplementedError('No verified Tensor output specification: '+name)
                    if list(values[name].shape)!=spec['shape'] or values[name].dtype!=getattr(torch,spec['dtype']):raise ValueError('Compiled output shape/dtype differs from package: '+name)
            result={name:values[name].cpu() if output_device=='cpu' else values[name] for name in self.config['outputs']}
        self.last_memory = lifetime.summary()
        return {'outputs':result,'seconds':time.perf_counter()-started,'compiler_calls':0,
                'logical_memory':self.last_memory,
                'npu_evidence':'requires_runtime_trace; device placement alone is not proof'}

    def memory_summary(self):
        return {'weight_logical_bytes':self.weight_logical_bytes,
                'execution':self.last_memory, 'physical_allocated_bytes':None}

    def close(self):
        stream = self.current_stream(self.torch.device('furiosa:0')) if callable(self.current_stream) else None
        sync = getattr(stream, 'synchronize', None)
        if callable(sync):
            sync()
        self.modules.clear()
        self.constants.clear()
