"""Native FXB transport and explicit execution of TCLMachine-owned tensor plans.

No compiler, model discovery, Python callbacks, chat scheduler or KV allocator is
loaded from a package. State updates are functional tensor outputs committed only
after stream completion. The producer must establish functional kernel inputs:
opaque EDF ABI alone cannot prove absence of in-place writes or storage aliases.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import tempfile
import threading
import time
from uuid import uuid4
import zipfile

from .graph_fxb import _bytes, _sha, _file_sha, _tensor_matches
from .native_fxb import MAX_JSON_SIZE, PRODUCER_MEMBER, NativeFXBError, build_native, validate_native


class ExecutionPlanError(NativeFXBError):
    pass


class RequestState:
    """A session-owned state loan, separate from shared model weights.

    Mirrors the public Rust Cache::lend ownership rule, using the installed
    Torch allocator. It is not a Rust HbmBuffer or a paged-attention engine.
    """
    def __init__(self,owner,tensors):
        self._owner=owner;self._tensors=tensors;self._closed=False;self._borrowed=False
        self.last_timing=None
    def close(self):
        if not self._closed:self._owner.release_state(self)
    def __enter__(self):return self
    def __exit__(self,kind,error,tb):
        try:self.close()
        except BaseException as cleanup:
            if error is not None:raise error.with_traceback(tb) from cleanup
            raise


def _last_uses(entry):
    return {name:index for index,step in enumerate(entry['steps']) for name in step['inputs']}


def _check_value_range(value,spec,torch):
    bounds=spec.get('value_range')
    if bounds is None:return
    if value.device.type!='cpu':
        raise ExecutionPlanError('Range-guarded inputs must be CPU tensors; implicit NPU-to-CPU validation is forbidden')
    with torch.no_grad():
        low,high=torch.aminmax(value)
        if low.item()<bounds['min'] or high.item()>=bounds['max_exclusive']:
            raise ExecutionPlanError('Tensor values are outside the declared value_range')


def _memory_gate(plan,*,clone_persistent_outputs=False):
    from .memory import tensor_bytes,check_budget
    specs=plan['tensors'];sizes={n:tensor_bytes(s) for n,s in specs.items()}
    persistent={n for n,s in specs.items() if s['kind'] in ('parameter','buffer','state')}
    # reset keeps the committed states alive until all replacement zeros finish.
    peak=sum(sizes[n] for n in persistent)+sum(sizes[n] for n in persistent if specs[n]['kind']=='state')
    for entry in plan['entries'].values():
        live=persistent|set(entry['inputs']);last=_last_uses(entry)
        retain=set(entry['outputs'])|set(entry['state_updates'].values())
        peak=max(peak,sum(sizes[n] for n in live))
        for index,step in enumerate(entry['steps']):
            live.update(step['outputs']);peak=max(peak,sum(sizes[n] for n in live))
            for name in step['inputs']+step['outputs']:
                if name not in persistent|retain and last.get(name,index)<=index:live.discard(name)
        if clone_persistent_outputs:
            exported_aliases=persistent|set(entry['state_updates'].values())
            peak=max(peak,sum(sizes[n] for n in live)+sum(sizes[n] for n in entry['outputs'] if n in exported_aliases))
    try:return check_budget(peak,plan.get('memory_policy'))
    except ValueError as exc:raise ExecutionPlanError(str(exc)) from exc


def validate_execution_plan(plan, *, archived=False):
    """Validate transport/dataflow, without deriving model semantics or compiling."""
    raw=_bytes(plan)
    if len(raw)>MAX_JSON_SIZE:raise ExecutionPlanError('Execution plan exceeds native JSON limit')
    plan=json.loads(raw)
    if not isinstance(plan,dict):raise ExecutionPlanError('Execution plan must be a mapping')
    def fail(message):raise ExecutionPlanError(message)
    def identifier(value):
        return isinstance(value,str) and 0<len(value)<=256 and not any(ord(c)<32 for c in value)
    def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    def spec(value):
        if not isinstance(value,dict):fail('Tensor spec must be a mapping')
        shape,stride=value.get('shape'),value.get('stride')
        if (not isinstance(shape,list) or not isinstance(stride,list) or len(shape)>16
                or len(shape)!=len(stride) or any(type(n)!=int or not 0<n<2**63 for n in shape)
                or any(type(n)!=int or not 0<=n<2**63 for n in stride)
                or not identifier(value.get('dtype'))):fail('Tensor requires concrete shape/dtype/stride')
        from .memory import tensor_bytes
        if tensor_bytes(value) is None:fail('Unknown tensor dtype')
    if plan.get('schema')!='tclmachine.execution-plan.v1':fail('Unsupported execution plan schema')
    if not isinstance(plan.get('source_metadata'),dict) or not identifier(plan['source_metadata'].get('architecture')):
        fail('Actual source architecture is required')
    if plan.get('device_config')!={'fusion':8,'mesh':'single'} or type(plan['device_config']['fusion'])!=int:fail('Only fusion8/Single is qualified')
    compatibility=plan.get('compatibility',{})
    if not isinstance(compatibility,dict):fail('SDK compatibility must be a mapping')
    packages=compatibility.get('packages',compatibility)
    if not isinstance(packages,dict) or any(not identifier(packages.get(p)) for p in ('torch','furiosa-torch','furiosa-tcc')):
        fail('Exact SDK package versions are required')
    if not isinstance(plan.get('compiler_identity'),dict) or any(not identifier(plan['compiler_identity'].get(k)) for k in ('version','revision')):
        fail('Compiler version/revision is required')
    tensors,kernels,entries=(plan.get(k) for k in ('tensors','kernels','entries'))
    if any(not isinstance(v,dict) or not v for v in (tensors,kernels,entries)):fail('Plan needs tensors, kernels and entries')
    for name,value in tensors.items():
        if not identifier(name):fail('Invalid tensor identifier')
        spec(value);kind=value.get('kind')
        if kind not in ('user_input','parameter','buffer','state','intermediate'):fail('Unsupported tensor kind')
        if kind in ('parameter','buffer') and not identifier(value.get('target')):fail('External state_dict target is required')
        if 'value_range' in value:
            bounds=value['value_range']
            if (kind not in ('user_input','parameter','buffer') or value['dtype'].removeprefix('torch.')!='int32'
                    or not isinstance(bounds,dict) or set(bounds)!={'min','max_exclusive'}
                    or any(type(bounds[k])!=int for k in ('min','max_exclusive'))
                    or not -(2**31)<=bounds['min']<bounds['max_exclusive']<=2**31):
                fail('value_range requires external int32 tensor and exact signed-i32 min/max_exclusive bounds')
        if kind=='state' and value.get('initial')!='zeros':fail('Only explicit zero-initialized state is supported')
        if kind=='state':
            stride=[];width=1
            for dimension in reversed(value['shape']):stride.insert(0,width);width*=dimension
            if value['stride']!=stride:fail('Zero state requires contiguous declared strides')
    for index,(name,kernel) in enumerate(kernels.items()):
        if not identifier(name) or not isinstance(kernel,dict):fail('Invalid kernel')
        if not digest(kernel.get('edf_sha256')) or type(kernel.get('expected_num_fusion'))!=int or kernel['expected_num_fusion']!=8:fail('Kernel needs EDF hash/fusion8')
        abi=kernel.get('abi',{})
        if not isinstance(abi,dict) or any(not isinstance(abi.get(k),list) or not abi[k] or any(not isinstance(x,str) or not x or len(x)>65536 for x in abi[k]) for k in ('inputs','outputs')):
            fail('Kernel exact EDF input/output ABI is required')
        if archived:
            member=kernel.get('member')
            if member!=f'edfs/kernel-{index:04d}.edf' or 'edf_path' in kernel:
                fail('Archived kernel must use its registered relative member')
        elif not isinstance(kernel.get('edf_path'),str):fail('Producer kernel needs an EDF path')
    signature={};used=set()
    def refs(value,label,unique=False):
        if not isinstance(value,list):fail(label+' must be a tensor-ID list')
        if any(not isinstance(x,str) or x not in tensors for x in value):fail(label+' references unknown tensors')
        if unique and len(value)!=len(set(value)):fail(label+' contains duplicate tensors')
        return value
    for name,entry in entries.items():
        if not identifier(name) or not isinstance(entry,dict):fail('Invalid entry')
        inputs=refs(entry.get('inputs'),'Entry inputs',True)
        if any(tensors[x]['kind']!='user_input' for x in inputs):fail('Entry inputs must be user_input tensors')
        available=set(inputs)|{n for n,s in tensors.items() if s['kind'] in ('parameter','buffer','state')}
        produced=set();steps=entry.get('steps')
        if not isinstance(steps,list) or not steps:fail('Entry needs ordered kernel steps')
        for step in steps:
            if not isinstance(step,dict) or not identifier(step.get('kernel')) or step['kernel'] not in kernels:fail('Unknown step kernel')
            key=step['kernel'];used.add(key);kernel=kernels[key]
            args=refs(step.get('inputs'),'Step inputs');out=refs(step.get('outputs'),'Step outputs',True)
            if any(x not in available for x in args):fail('Step reads a tensor before it is available')
            if len(args)!=len(kernel['abi']['inputs']) or len(out)!=len(kernel['abi']['outputs']):fail('Step/EDF arity mismatch')
            if any(x in available or tensors[x]['kind']!='intermediate' for x in out):fail('Step outputs cannot alias or overwrite live inputs/state')
            logical=tuple(tuple((k,tuple(tensors[x][k]) if k!='dtype' else tensors[x][k]) for k in ('shape','dtype','stride')) for x in args+out)
            if key in signature and signature[key]!=logical:fail('Kernel reused with different logical tensor contracts')
            signature[key]=logical;available.update(out);produced.update(out)
        outputs=refs(entry.get('outputs'),'Entry outputs',True)
        if not outputs or any(x not in available for x in outputs):fail('Entry outputs must be available tensors')
        updates=entry.get('state_updates')
        if not isinstance(updates,dict):fail('Entry needs explicit state_updates mapping')
        for target,value in updates.items():
            if target not in tensors or tensors[target]['kind']!='state' or not isinstance(value,str) or value not in produced:
                fail('State update must bind a state slot to a produced tensor')
            if any(tensors[target][k]!=tensors[value][k] for k in ('shape','dtype','stride')):
                fail('State update shape/dtype/stride mismatch')
    if used!=set(kernels):fail('Unreferenced kernels are not allowed')
    _memory_gate(plan)
    return plan


def export_execution_plan(plan, output, *, verify_official=True):
    """Package a producer's qualified tensor plan; do not discover/compile kernels."""
    plan=validate_execution_plan(plan)
    staged_plan=json.loads(_bytes(plan));paths={}
    for index,(name,kernel) in enumerate(staged_plan['kernels'].items()):
        source=Path(kernel.pop('edf_path'))
        if source.is_symlink() or not source.is_file() or source.suffix!='.edf' or _file_sha(source)!=kernel['edf_sha256']:
            raise ExecutionPlanError('Expected unchanged genuine EDF producer artifact')
        member=f'edfs/kernel-{index:04d}.edf';kernel['member']=member;paths[member]=(source,kernel['edf_sha256'])
    validate_execution_plan(staged_plan,archived=True)
    fingerprint=_sha(staged_plan)
    manifest={'format_version':2,'uuid':str(uuid4()),'created_at':datetime.now(timezone.utc).isoformat(),
        'versions':{'furiosa_compiler':plan['compiler_identity']},
        'model':{'architecture':plan['source_metadata']['architecture'],'fxbx_execution_fingerprint':fingerprint},
        'parallel_config':{'tensor_parallel_size':8,'pipeline_parallel_size':1},'kernels':{}}
    for name,kernel in staged_plan['kernels'].items():
        manifest['kernels'][name]=[{'bucket':{'tokenwise':{'input_size':1}},'file':kernel['member']}]
    with tempfile.TemporaryDirectory(prefix='fxbx-execution-') as directory:
        root=Path(directory);(root/'manifest.json').write_bytes(_bytes(manifest))
        for member,(source,expected) in paths.items():
            target=root/member;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(source,target)
            if _file_sha(target)!=expected:
                raise ExecutionPlanError('EDF changed during staging')
        result=build_native(root,output,verify_official=verify_official,
            producer_metadata={'execution_plan':staged_plan,'execution_plan_sha256':fingerprint,
                               'official_llm_dispatch':False,'explicit_consumer_required':True})
    result['gates']['execution_plan']='passed';return result


def _archive_plan(archive):
    """Read the binding from the same open archive whose EDFs will be loaded."""
    def document(name):
        try:
            if archive.getinfo(name).file_size>MAX_JSON_SIZE:raise ExecutionPlanError('Native JSON exceeds size limit')
            return json.loads(archive.read(name))
        except (KeyError,ValueError,UnicodeError,zipfile.BadZipFile) as exc:
            if isinstance(exc,ExecutionPlanError):raise
            raise ExecutionPlanError('Invalid execution archive document: '+name) from exc
    producer=document(PRODUCER_MEMBER)
    if not isinstance(producer,dict) or producer.get('producer')!='fxbex':raise ExecutionPlanError('Missing fxbX execution producer')
    plan=validate_execution_plan(producer.get('execution_plan'),archived=True)
    digest=_sha(plan)
    if producer.get('execution_plan_sha256')!=digest:raise ExecutionPlanError('Execution plan hash changed')
    manifest=document('manifest.json')
    expected={name:[{'bucket':{'tokenwise':{'input_size':1}},'file':k['member']}] for name,k in plan['kernels'].items()}
    members=[info.filename for info in archive.infolist()]
    if len(members)!=len(set(members)) or set(members)!={'manifest.json',PRODUCER_MEMBER}|{k['member'] for k in plan['kernels'].values()}:
        raise ExecutionPlanError('Archive membership differs from execution plan')
    if (not isinstance(manifest,dict) or manifest.get('format_version')!=2
            or manifest.get('model')!={'architecture':plan['source_metadata']['architecture'],'fxbx_execution_fingerprint':digest}
            or manifest.get('parallel_config')!={'tensor_parallel_size':8,'pipeline_parallel_size':1}
            or manifest.get('versions')!={'furiosa_compiler':plan['compiler_identity']} or manifest.get('kernels')!=expected):
        raise ExecutionPlanError('Native manifest differs from execution plan')
    return plan


def read_execution_plan(path):
    validate_native(path,verify_official=False)
    with zipfile.ZipFile(path) as archive:
        plan=_archive_plan(archive)
        for kernel in plan['kernels'].values():
            with archive.open(kernel['member']) as stream:
                h=hashlib.sha256()
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            if h.hexdigest()!=kernel['edf_sha256']:raise ExecutionPlanError('EDF hash changed')
    return plan


class ExecutionFXB:
    """Exclusive fixed-shape tensor-plan session; state and temporaries stay on NPU."""
    def __init__(self,path,*,weights=None,device='furiosa:0',max_request_states=1):
        if device!='furiosa:0':raise ExecutionPlanError('Only furiosa:0 is qualified')
        if type(max_request_states)!=int or max_request_states<1:raise ExecutionPlanError('max_request_states must be a positive integer')
        from .checkpoint import SafeTensorWeights
        self.path=Path(path);self.plan=read_execution_plan(path)
        self.weights=weights if isinstance(weights,SafeTensorWeights) else dict(weights or {})
        self.device=device;self.programs={};self.resident={};self.states={};self.stack=None;self.failed=False
        self.memory=_memory_gate(self.plan)
        self._memory_modes={'cpu':self.memory}
        self._lock=threading.RLock()
        self._requests=set();self.last_timing=None;self.max_request_states=max_request_states;self._default_ready=False
        from .memory import tensor_bytes
        self._state_bytes=sum(tensor_bytes(t) for t in self.plan['tensors'].values() if t['kind']=='state')
    def __enter__(self):
        with self._lock:
            if self.stack is not None:raise ExecutionPlanError('Session already open')
            self.stack=ExitStack()
            try:self._load()
            except BaseException as error:
                try:self.close()
                except BaseException as cleanup:raise error from cleanup
                raise
            return self
    def _load(self):
        import torch,furiosa.torch as sdk
        from furiosa.torch.config import set_cpu_fallback,get_cpu_fallback
        packages=self.plan['compatibility'];packages=packages.get('packages',packages)
        for name,expected in packages.items():
            if name in ('torch','furiosa-torch','furiosa-tcc') and importlib.metadata.version(name)!=expected:
                raise ExecutionPlanError('SDK package mismatch: '+name)
        from .checkpoint import SafeTensorWeights
        streaming=isinstance(self.weights,SafeTensorWeights)
        if streaming:
            targets={}
            for spec in self.plan['tensors'].values():
                if spec['kind'] not in ('parameter','buffer'):continue
                target=spec['target']
                if target in targets and any(targets[target].get(k)!=spec.get(k) for k in ('shape','stride','dtype','value_range')):
                    raise ExecutionPlanError('External target contracts disagree: '+target)
                targets[target]=spec
            try:self.weights.validate(targets)
            except ValueError as exc:raise ExecutionPlanError(str(exc)) from exc
            # Integer range contracts need value checks before device allocation.
            for target,spec in targets.items():
                if 'value_range' in spec:_check_value_range(self.weights[target],spec,torch)
        for name,spec in self.plan['tensors'].items():
            if spec['kind'] in ('parameter','buffer'):
                if streaming:continue
                value=self.weights.get(spec['target'])
                if not _tensor_matches(value,spec,torch) or value.requires_grad:
                    raise ExecutionPlanError('Missing/incompatible frozen external tensor: '+name)
                _check_value_range(value,spec,torch)
        self.torch,self.sdk=torch,sdk
        sdk.set_fusion(8);sdk.set_mesh_kind(sdk.MeshKind.Single());runnables={}
        with zipfile.ZipFile(self.path) as archive:
            if _archive_plan(archive)!=self.plan:raise ExecutionPlanError('Execution plan changed after session construction')
            for name,k in self.plan['kernels'].items():
                raw=archive.read(k['member'])
                if hashlib.sha256(raw).hexdigest()!=k['edf_sha256']:raise ExecutionPlanError('EDF changed before loading')
                program=sdk.ir.Runnable.from_serialized_edf(raw)
                abi={'inputs':[str(x) for x in program.input_dram_shapes],'outputs':[str(x) for x in program.output_dram_shapes]}
                if any(abi[d]!=k['abi'][d] for d in ('inputs','outputs')) or getattr(program,'expected_num_fusion',None)!=8:raise ExecutionPlanError('Official EDF ABI/fusion mismatch')
                runnables[name]=program
        if not callable(getattr(sdk,'current_stream',None)):raise ExecutionPlanError('Public completion API unavailable')
        from .settings import npu_lock
        self.stack.enter_context(npu_lock());set_cpu_fallback(False)
        if get_cpu_fallback():raise ExecutionPlanError('CPU fallback could not be disabled')
        for name,program in runnables.items():self.programs[name]=sdk.Runnable(program).to(self.device)
        uploaded={}
        for name,spec in self.plan['tensors'].items():
            if spec['kind'] in ('parameter','buffer'):
                target=spec['target']
                if target not in uploaded:
                    value=self.weights[target]
                    if not _tensor_matches(value,spec,torch) or value.requires_grad:
                        raise ExecutionPlanError('External tensor changed after admission: '+target)
                    self.resident[name]=self._upload(value.detach(),spec);uploaded[target]=name
                    if streaming:self._sync()
                    del value
                else:self.resident[name]=self.resident[uploaded[target]]
        self._sync();self.states={};self._default_ready=False;self.failed=False
    def _sync(self):self.sdk.current_stream(self.torch.device(self.device)).synchronize()
    def _upload(self,value,spec):
        result=value.to(self.device)
        if not _tensor_matches(result,spec,self.torch) or result.device.type!='furiosa':
            raise ExecutionPlanError('Tensor upload did not preserve its NPU contract')
        return result
    def _allocate_states(self):
        states={}
        for name,spec in self.plan['tensors'].items():
            if spec['kind']=='state':
                dtype=getattr(self.torch,spec['dtype'].removeprefix('torch.'),None)
                if dtype is None:raise ExecutionPlanError('Unknown state dtype')
                value=self.torch.zeros(spec['shape'],dtype=dtype)
                if not _tensor_matches(value,spec,self.torch):raise ExecutionPlanError('Zero state needs its declared contiguous layout')
                states[name]=self._upload(value,spec)
        self._sync();return states
    def _reset(self):self.states=self._allocate_states();self._default_ready=True
    def _check_state(self,state):
        if (not isinstance(state,RequestState) or state._owner is not self or state._closed or state not in self._requests):
            raise ExecutionPlanError('Request state belongs to another or closed session')
        if state._borrowed:raise ExecutionPlanError('Request state is already borrowed by a call')
    def _admit_requests(self,count,output_device='cpu',*,default_present=None):
        if output_device not in self._memory_modes:self._memory_modes[output_device]=_memory_gate(self.plan,clone_persistent_outputs=True)
        default_present=self._default_ready if default_present is None else default_present
        result=dict(self._memory_modes[output_device]);result['known_required_bytes']+=max(0,count+int(default_present)-1)*self._state_bytes
        if result['budget_bytes'] is not None and result['known_required_bytes']>result['budget_bytes']:
            raise ExecutionPlanError('Logical memory estimate including request states exceeds configured budget')
        return result
    def new_state(self):
        with self._lock:
            self._healthy()
            if not self._state_bytes:raise ExecutionPlanError('Plan has no state slots')
            if len(self._requests)>=self.max_request_states:raise ExecutionPlanError('Request state limit reached; release an existing request')
            memory=self._admit_requests(len(self._requests)+1)
            try:state=RequestState(self,self._allocate_states())
            except BaseException:self.failed=True;raise
            self._requests.add(state);self.memory=memory;return state
    def release_state(self,state):
        with self._lock:
            self._check_state(state)
            try:self._sync()
            except BaseException:self.failed=True;raise
            state._tensors.clear();state._closed=True
            self._requests.remove(state);self.memory=self._admit_requests(len(self._requests))
    def reset(self,*,state=None):
        with self._lock:
            self._healthy()
            if state is not None:self._check_state(state)
            self._admit_requests(len(self._requests),default_present=self._default_ready or state is None)
            try:
                if state is None:self._reset()
                else:state._tensors=self._allocate_states()
            except BaseException:self.failed=True;raise
    def _healthy(self):
        if self.stack is None or not self.programs or self.failed:raise ExecutionPlanError('Session must be loaded and healthy')
    def predict(self,inputs=(),*,output_device='cpu',state=None):
        return self.run('forward',inputs,output_device=output_device,state=state)
    def run(self,entry,inputs=(),*,output_device='cpu',state=None):
        queued=time.perf_counter()
        with self._lock:
            wait_s=time.perf_counter()-queued;started=time.perf_counter()
            self._healthy()
            if not isinstance(entry,str) or entry not in self.plan['entries'] or output_device not in ('cpu','furiosa'):raise ExecutionPlanError('Unknown entry/output device')
            if state is not None:self._check_state(state)
            self.memory=self._admit_requests(len(self._requests),output_device,default_present=self._default_ready or state is None)
            recipe=self.plan['entries'][entry]
            last=_last_uses(recipe);retain=set(recipe['outputs'])|set(recipe['state_updates'].values())
            if isinstance(inputs,dict):
                if set(inputs)!=set(recipe['inputs']):raise ExecutionPlanError('Entry input names differ')
                values=inputs
            elif isinstance(inputs,(list,tuple)) and len(inputs)==len(recipe['inputs']):values=dict(zip(recipe['inputs'],inputs))
            else:raise ExecutionPlanError('Entry input arity mismatch')
            for name,value in values.items():
                if not _tensor_matches(value,self.plan['tensors'][name],self.torch) or value.requires_grad:
                    raise ExecutionPlanError('Entry input metadata mismatch: '+name)
                _check_value_range(value,self.plan['tensors'][name],self.torch)
            if state is not None:state._borrowed=True
            try:
                with self.torch.no_grad():
                    if state is None and not self._default_ready:self._reset()
                    active=self.states if state is None else state._tensors
                    tensors={**self.resident,**active,**{k:self._upload(v,self.plan['tensors'][k]) for k,v in values.items()}}
                    for index,step in enumerate(recipe['steps']):
                        marker=nullcontext()
                        if os.environ.get('FXBX_TRACE_STEPS')=='1':
                            from furiosa.torch.profiler import RecordFunction
                            marker=RecordFunction(f"execution_plan::{entry}::{index}::{step['kernel']}",propagate=True)
                        with marker:result=self.programs[step['kernel']](*(tensors[n] for n in step['inputs']))
                        if not isinstance(result,(tuple,list)):result=[result]
                        if len(result)!=len(step['outputs']):raise ExecutionPlanError('Kernel output arity mismatch')
                        for name,value in zip(step['outputs'],result):
                            if (not _tensor_matches(value,self.plan['tensors'][name],self.torch)
                                    or value.device.type!='furiosa'):raise ExecutionPlanError('NPU output metadata mismatch: '+name)
                            tensors[name]=value
                        del result,value
                        for name in step['inputs']+step['outputs']:
                            if (self.plan['tensors'][name]['kind'] in ('user_input','intermediate')
                                    and name not in retain and last.get(name,index)<=index):tensors.pop(name,None)
                    persistent_outputs=set(self.resident)|set(active)|set(recipe['state_updates'].values())
                    output=tuple(tensors[n].cpu() if output_device=='cpu' else
                                 tensors[n].clone() if n in persistent_outputs else tensors[n]
                                 for n in recipe['outputs'])
                    self._sync()
                    committed={**active,**{slot:tensors[value] for slot,value in recipe['state_updates'].items()}}
                    if state is None:self.states=committed
                    else:state._tensors=committed
                    self.last_timing={'entry':entry,'queue_wait_s':wait_s,'request_s':time.perf_counter()-started,'scope':'completed host request; queue wait excluded'}
                    if state is not None:state.last_timing=dict(self.last_timing)
                    return output
            except BaseException:self.failed=True;raise
            finally:
                if state is not None:state._borrowed=False
    def close(self):
        with self._lock:
            try:
                if self.programs:self._sync()
            finally:
                for state in self._requests:state._tensors.clear();state._closed=True
                self._requests.clear()
                self.programs.clear();self.resident.clear();self.states.clear();self._default_ready=False
                if self.stack is not None:self.stack.close();self.stack=None
    def __exit__(self,kind,error,tb):
        try:self.close()
        except BaseException as cleanup:
            if error is not None:raise error.with_traceback(tb) from cleanup
            raise
