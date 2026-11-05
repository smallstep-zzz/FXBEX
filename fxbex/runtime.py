"""Fixed-plan runtime. Only built-in, installed adapters may execute a package."""
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
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


def package_compatibility(manifest):
    actual = environment(); notices=[]
    import sys
    if (manifest.get('model',{}).get('name')=='RedHatAI/Qwen3.5-9B-FP8-dynamic'
            and sys.version_info[:2]==(3,11) and actual.get('furiosa-torch')=='2026.4.0'
            and actual.get('torch')=='2.10.0+cpu' and actual.get('transformers')=='5.22.1'):
        raise PackageError('Known unstable Qwen runtime combination: Python 3.11, SDK 2026.4.0, transformers 5.22.1 caused native crashes. Use the validated transformers==5.20.2 in this environment.')
    for name, expected in manifest['compatibility'].get('packages', {}).items():
        installed=actual.get(name)
        if installed == expected:continue
        if (name=='transformers' and isinstance(installed,str) and isinstance(expected,str)
                and installed.split('.')[0].isdigit() and installed.split('.')[0]==expected.split('.')[0]):
            notices.append(f'transformers differs: package={expected}, installed={installed}; proceeding with adapter validation; compatibility is not yet established')
        else:
            raise PackageError(f'SDK mismatch {name}: package={expected}, installed={installed}; rebuild explicitly')
    return notices


def check_compatibility(manifest):
    import warnings
    for message in package_compatibility(manifest):warnings.warn(message,RuntimeWarning,stacklevel=2)
    if manifest['compatibility'].get('hardware', 'rngd') != 'rngd' or manifest['compatibility'].get('fusion',8) != 8:
        raise PackageError('This adapter version requires RNGD fusion8')
    from .bundled import specification
    if specification(manifest) is None:resolve_adapter(manifest['runtime']['adapter'])


class Session:
    def __init__(self, path, *, report_dir=None, weights_dir=None):
        self.path = Path(path)
        self.weights_dir = Path(weights_dir).resolve() if weights_dir else None
        self.stack = ExitStack()
        self.report_dir = Path(report_dir or runs_path()/f'fxbex_{time.time_ns()}').resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = None
        self.telemetry = None
        self.memory = None

    def __enter__(self):
        try:
            self.root = self.stack.enter_context(extract(self.path))
            self.manifest = json.loads((self.root/'manifest.json').read_text())
            check_compatibility(self.manifest)
            from .memory import analyze
            memory_plan = analyze(self.root)
            (self.report_dir/'memory_plan.json').write_text(json.dumps(memory_plan, indent=2))
            from .execution import ExecutionRecorder, MemoryTelemetry
            self.execution = ExecutionRecorder(self.report_dir, self.manifest)
            self.telemetry = MemoryTelemetry(self.report_dir)
            self.memory = self.telemetry.sample('before_load')
            # Share the canonical lock with web workers. Never assume PE partitioning.
            import fcntl
            canonical = lock_path()
            canonical.parent.mkdir(parents=True, exist_ok=True)
            lock = self.stack.enter_context(canonical.open('a'))
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        self.sample_memory(f'before_{operation}')
        try:
            return self.execution.call(operation, callback)
        finally:
            self.sample_memory(f'after_{operation}')

    def predict(self, inputs):
        if not hasattr(self.adapter, 'predict'):raise ValueError('Package does not provide predict')
        return self._call('predict', lambda: self.adapter.predict(inputs))

    def chat(self, messages, max_new_tokens=32):
        if not hasattr(self.adapter, 'chat'): raise ValueError('Package does not provide chat')
        return self._call('chat', lambda: self.adapter.chat(messages, max_new_tokens))


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
        self.steps = plan['steps']
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
            artifact = root/step['artifact']
            if not artifact.resolve().is_relative_to(root.resolve()): raise PackageError('Artifact escapes package')
            self.modules[step['id']] = Runnable(ir.Runnable.deserialize(artifact.read_bytes())).to('furiosa:0')
            known.update(step['outputs'])
        if any(name not in known for name in self.config['outputs']): raise PackageError('Unbound DAG output')

    def predict(self, inputs):
        from .memory import TensorLifetime
        torch=self.torch
        if set(inputs) != set(self.config['inputs']): raise ValueError('Input names do not match package')
        values=dict(self.constants)
        for name, spec in self.config['inputs'].items():
            tensor=torch.tensor(inputs[name], dtype=getattr(torch,spec['dtype']))
            if list(tensor.shape)!=spec['shape']: raise ValueError(f'Input shape mismatch: {name}')
            values[name]=tensor.to('furiosa:0')
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
            result={name:values[name].cpu().float().tolist() for name in self.config['outputs']}
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
