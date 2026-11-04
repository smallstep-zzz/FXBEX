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


def check_compatibility(manifest):
    actual = environment()
    for name, expected in manifest['compatibility'].get('packages', {}).items():
        if actual.get(name) != expected:
            raise PackageError(f'SDK mismatch {name}: package={expected}, installed={actual.get(name)}; rebuild explicitly')
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

    def __enter__(self):
        try:
            self.root = self.stack.enter_context(extract(self.path))
            self.manifest = json.loads((self.root/'manifest.json').read_text())
            check_compatibility(self.manifest)
            from .execution import ExecutionRecorder
            self.execution = ExecutionRecorder(self.report_dir, self.manifest)
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
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *args):
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

    def predict(self, inputs):
        if not hasattr(self.adapter, 'predict'):raise ValueError('Package does not provide predict')
        return self.execution.call('predict', lambda: self.adapter.predict(inputs))

    def chat(self, messages, max_new_tokens=32):
        if not hasattr(self.adapter, 'chat'): raise ValueError('Package does not provide chat')
        return self.execution.call('chat', lambda: self.adapter.chat(messages, max_new_tokens))


class RunnableDAG:
    """Typed ordered DAG with input/weight/state/output bindings; no model CPU ops."""
    def __init__(self, root, weights_dir=None):
        import torch
        from furiosa.torch import ir, Runnable
        from furiosa.torch import set_fusion,set_mesh_kind,MeshKind
        from furiosa.torch.config import set_cpu_fallback,get_cpu_fallback
        set_cpu_fallback(False)
        if get_cpu_fallback():raise RuntimeError('Cannot disable CPU fallback')
        set_fusion(8);set_mesh_kind(MeshKind.Single())
        self.torch = torch
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
            for name, tensor in tensors.items(): self.constants[name]=tensor.to('furiosa:0')
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
        torch=self.torch
        if set(inputs) != set(self.config['inputs']): raise ValueError('Input names do not match package')
        values=dict(self.constants)
        for name, spec in self.config['inputs'].items():
            tensor=torch.tensor(inputs[name], dtype=getattr(torch,spec['dtype']))
            if list(tensor.shape)!=spec['shape']: raise ValueError(f'Input shape mismatch: {name}')
            values[name]=tensor.to('furiosa:0')
        started=time.perf_counter()
        with torch.no_grad():
            for step in self.steps:
                outputs=self.modules[step['id']](*(values[n] for n in step['inputs']))
                if len(outputs)!=len(step['outputs']): raise PackageError('Kernel output arity mismatch')
                values.update(zip(step['outputs'],outputs))
            result={name:values[name].cpu().float().tolist() for name in self.config['outputs']}
        return {'outputs':result,'seconds':time.perf_counter()-started,'compiler_calls':0,
                'npu_evidence':'requires_runtime_trace; device placement alone is not proof'}
