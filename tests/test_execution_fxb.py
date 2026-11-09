"""Offline execution-contract tests; fake SDK never establishes NPU execution."""
from contextlib import nullcontext
import copy,hashlib,json,sys,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest.mock import patch

from fxbex.execution_fxb import (ExecutionFXB,ExecutionPlanError,export_execution_plan,
                                  read_execution_plan,validate_execution_plan,_check_value_range)
from fxbex.native_fxb import PRODUCER_MEMBER


class Tensor:
    def __init__(self,value,device='cpu'):
        self.value=value;self.device=types.SimpleNamespace(type=device.split(':')[0]);self.shape=(1,2)
        self.dtype='torch.float32';self.requires_grad=False
    def stride(self):return (2,1)
    def detach(self):return self
    def to(self,device):return Tensor(self.value,device)
    def cpu(self):return self.to('cpu')
    def clone(self):return Tensor(self.value,self.device.type)
    def item(self):return self.value


class ExecutionPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.output=self.root/'plan.fxb'
        kernels={}
        for name,args in (('add',['x ABI','state ABI']),('double',['middle ABI'])):
            path=self.root/(name+'.edf');path.write_bytes(name.encode())
            kernels[name]={'edf_path':str(path),'edf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'abi':{'inputs':args,'outputs':['middle ABI' if name=='add' else 'result ABI']},'expected_num_fusion':8}
        spec={'shape':[1,2],'stride':[2,1],'dtype':'float32'}
        self.plan={'schema':'tclmachine.execution-plan.v1','source_metadata':{'architecture':'OfflineStatefulFixture'},
            'compatibility':{'torch':'test','furiosa-torch':'test','furiosa-tcc':'test'},
            'compiler_identity':{'version':'test','revision':'fixture'},'device_config':{'fusion':8,'mesh':'single'},
            'tensors':{'x':dict(spec,kind='user_input'),'state':dict(spec,kind='state',initial='zeros'),
                       'a':dict(spec,kind='intermediate'),'y':dict(spec,kind='intermediate')},
            'kernels':kernels,'entries':{'forward':{'inputs':['x'],'steps':[
                {'kernel':'add','inputs':['x','state'],'outputs':['a']},
                {'kernel':'double','inputs':['a'],'outputs':['y']}],'outputs':['y'],'state_updates':{'state':'y'}}}}
    def export(self):return export_execution_plan(self.plan,self.output,verify_official=False)
    def test_native_manifest_has_each_edf_and_no_absolute_kernel_path(self):
        result=self.export();plan=read_execution_plan(self.output)
        self.assertEqual(result['gates']['npu_execution'],'not_run')
        self.assertEqual(len(plan['kernels']),2)
        self.assertTrue(all('edf_path' not in k and k['member'].startswith('edfs/') for k in plan['kernels'].values()))
        with zipfile.ZipFile(self.output) as archive:
            manifest=json.loads(archive.read('manifest.json'))
            self.assertEqual(set(manifest['kernels']),{'add','double'})
            self.assertEqual(manifest['model']['architecture'],'OfflineStatefulFixture')
        with self.assertRaises(FileExistsError):self.export()
    def test_bad_dataflow_and_state_contracts_are_rejected(self):
        changes=[lambda p:p['entries']['forward']['steps'][0]['inputs'].__setitem__(0,'missing'),
                 lambda p:p['entries']['forward']['steps'][0]['inputs'].__setitem__(0,'y'),
                 lambda p:p['entries']['forward']['steps'][0]['outputs'].__setitem__(0,'state'),
                 lambda p:p['entries']['forward']['steps'][0]['outputs'].__setitem__(0,'x'),
                 lambda p:p['entries']['forward']['state_updates'].__setitem__('x','y'),
                 lambda p:p['entries']['forward']['state_updates'].__setitem__('state','x'),
                 lambda p:p['tensors']['y'].__setitem__('shape',[2,2]),
                 lambda p:p['entries']['forward'].__setitem__('inputs',['x','x']),
                 lambda p:p['entries']['forward']['steps'][0].__setitem__('inputs',[{}]),
                 lambda p:p['entries']['forward']['steps'][0].__setitem__('kernel',{}),
                 lambda p:p['entries']['forward']['state_updates'].__setitem__('state',{}),
                 lambda p:p['kernels']['double'].__setitem__('abi',[]),
                 lambda p:p['kernels']['double']['abi'].__setitem__('inputs',['one','two'])]
        for change in changes:
            with self.subTest(change=change):
                p=copy.deepcopy(self.plan);change(p)
                with self.assertRaises(ExecutionPlanError):validate_execution_plan(p)
    def test_scalar_specs_are_supported_and_missing_plan_is_actionable(self):
        for spec in self.plan['tensors'].values():spec.update(shape=[],stride=[])
        validate_execution_plan(self.plan)
        for value in (None,[],1):
            with self.assertRaises(ExecutionPlanError):validate_execution_plan(value)
    def test_value_range_contract_is_external_int32_only(self):
        spec=self.plan['tensors']['x'];spec.update(dtype='int32',value_range={'min':0,'max_exclusive':8})
        validate_execution_plan(self.plan)
        for bounds in ({'min':0,'max_exclusive':0},{'min':False,'max_exclusive':8},
                       {'min':0.,'max_exclusive':8},{'min':0,'max_exclusive':2**31+1},
                       {'min':-(2**31)-1,'max_exclusive':8},{'min':0,'max_exclusive':8,'extra':1}):
            p=copy.deepcopy(self.plan);p['tensors']['x']['value_range']=bounds
            with self.assertRaisesRegex(ExecutionPlanError,'value_range'):validate_execution_plan(p)
        for name in ('state','a'):
            p=copy.deepcopy(self.plan);p['tensors'][name].update(dtype='int32',value_range={'min':0,'max_exclusive':8})
            with self.assertRaisesRegex(ExecutionPlanError,'value_range'):validate_execution_plan(p)
    def test_range_guard_never_copies_device_values_to_cpu(self):
        torch=types.SimpleNamespace(no_grad=nullcontext,aminmax=lambda v:(Tensor(v.value),Tensor(v.value)))
        spec={'value_range':{'min':0,'max_exclusive':8}}
        _check_value_range(Tensor(0),spec,torch);_check_value_range(Tensor(7),spec,torch)
        for value in (Tensor(-1),Tensor(8),Tensor(1,'furiosa:0')):
            with self.assertRaises(ExecutionPlanError):_check_value_range(value,spec,torch)
    def test_invalid_external_indices_do_not_launch_or_poison_session(self):
        self.plan['tensors']['x'].update(dtype='int32',value_range={'min':0,'max_exclusive':8})
        self.export();modules,calls=self.backend()
        modules['torch'].aminmax=lambda value:(Tensor(value.value),Tensor(value.value))
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(self.output) as model:
                value=Tensor(8);value.dtype='torch.int32'
                with self.assertRaisesRegex(ExecutionPlanError,'value_range'):model.predict((value,))
                value=Tensor(1,'furiosa:0');value.dtype='torch.int32'
                with self.assertRaisesRegex(ExecutionPlanError,'CPU tensors'):model.predict((value,))
                self.assertFalse(model.failed)
                self.assertFalse(any(c[0]=='run' for c in calls))
    def test_invalid_range_guarded_weights_fail_before_device_allocation(self):
        self.plan['tensors']['indices']={'shape':[1,2],'stride':[2,1],'dtype':'int32','kind':'buffer','target':'indices','value_range':{'min':0,'max_exclusive':8}}
        self.export();modules,calls=self.backend();modules['torch'].aminmax=lambda value:(Tensor(value.value),Tensor(value.value))
        value=Tensor(-1);value.dtype='torch.int32'
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'):
            with self.assertRaisesRegex(ExecutionPlanError,'value_range'):
                with ExecutionFXB(self.output,weights={'indices':value}):pass
        self.assertFalse(any(c[0]=='load' for c in calls))
    def test_memory_budget_is_explicit_and_conservative(self):
        self.plan['memory_policy']={'budget_bytes':1}
        with self.assertRaisesRegex(ExecutionPlanError,'Logical memory estimate'):self.export()
    def test_reset_peak_includes_old_and_replacement_state(self):
        self.plan['tensors']['state'].update(shape=[1024],stride=[1])
        self.plan['entries']['forward']['steps'][0]['inputs']=['x','x']
        self.plan['entries']['forward']['state_updates']={}
        self.plan['memory_policy']={'budget_bytes':8191}
        with self.assertRaisesRegex(ExecutionPlanError,'8192'):self.export()
    def test_edf_corruption_and_changed_plan_are_rejected(self):
        Path(self.plan['kernels']['add']['edf_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ExecutionPlanError,'unchanged'):self.export()
        self.assertFalse(self.output.exists())
    def test_manifest_or_hash_tampering_is_rejected(self):
        self.export()
        with zipfile.ZipFile(self.output) as z:files={n:z.read(n) for n in z.namelist()}
        meta=json.loads(files[PRODUCER_MEMBER]);meta['execution_plan']['entries']['forward']['outputs']=['a']
        files[PRODUCER_MEMBER]=json.dumps(meta).encode()
        with zipfile.ZipFile(self.output,'w') as z:
            for n,v in files.items():z.writestr(n,v)
        with self.assertRaisesRegex(ExecutionPlanError,'hash changed'):read_execution_plan(self.output)

    def backend(self,*,bad_abi=False,fail_step=False):
        calls=[];state={'fusion':1,'fallback':True};torch=types.ModuleType('torch');torch.Tensor=Tensor
        torch.device=lambda x:x;torch.no_grad=nullcontext;torch.float32='torch.float32'
        torch.zeros=lambda shape,dtype:Tensor(0)
        sdk=types.ModuleType('furiosa.torch')
        def deserialize(raw):
            calls.append(('preprocess',state['fusion']))
            name=raw.decode();args=self.plan['kernels'][name]['abi']
            return types.SimpleNamespace(name=name,input_dram_shapes=['BAD'] if bad_abi else args['inputs'],
                output_dram_shapes=args['outputs'],expected_num_fusion=state['fusion'])
        sdk.ir=types.SimpleNamespace(Runnable=types.SimpleNamespace(from_serialized_edf=deserialize))
        class Program:
            def __init__(self,ir):self.name=ir.name
            def to(self,device):calls.append(('load',self.name));return self
            def __call__(self,*args):
                calls.append(('run',self.name,[a.device.type for a in args]))
                if fail_step and self.name=='double':raise RuntimeError('kernel failure')
                value=args[0].value+args[1].value if self.name=='add' else args[0].value*2
                return [Tensor(value,'furiosa:0')]
        sdk.Runnable=Program
        def fusion(value):state['fusion']=value;calls.append(('fusion',value))
        sdk.set_fusion=fusion;sdk.set_mesh_kind=lambda value:None;sdk.MeshKind=types.SimpleNamespace(Single=lambda:'single')
        sdk.current_stream=lambda device:types.SimpleNamespace(synchronize=lambda:calls.append(('sync',)))
        config=types.ModuleType('furiosa.torch.config');config.set_cpu_fallback=lambda value:state.update(fallback=value)
        config.get_cpu_fallback=lambda:state['fallback'];furiosa=types.ModuleType('furiosa');furiosa.torch=sdk
        return {'torch':torch,'furiosa':furiosa,'furiosa.torch':sdk,'furiosa.torch.config':config},calls
    def test_runtime_multistep_state_reset_and_completion(self):
        self.export();modules,calls=self.backend()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(self.output) as model:
                self.assertEqual(model.predict((Tensor(1),))[0].value,2)
                self.assertEqual(model.run('forward',{'x':Tensor(1)})[0].value,6)
                self.assertEqual(model.states['state'].device.type,'furiosa')
                model.reset();self.assertEqual(model.predict((Tensor(1),))[0].value,2)
            with self.assertRaises(ExecutionPlanError):model.predict((Tensor(1),))
        self.assertLess(calls.index(('fusion',8)),calls.index(('preprocess',8)))
        self.assertTrue(all(all(d=='furiosa' for d in call[2]) for call in calls if call[0]=='run'))
    def test_gpu_state_output_does_not_expose_mutable_committed_slot(self):
        self.export();modules,calls=self.backend()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(self.output) as model:
                output,=model.predict((Tensor(1),),output_device='furiosa')
                self.assertIsNot(output,model.states['state'])
                output.value=999
                self.assertEqual(model.states['state'].value,2)
                self.assertEqual(model.predict((Tensor(1),))[0].value,6)
    def test_gpu_state_output_clone_is_in_memory_admission(self):
        # CPU route fits24 bytes; returning a separate GPU state result needs32.
        self.plan['entries']['forward']['outputs']=['a','y']
        self.plan['memory_policy']={'budget_bytes':31}
        self.export();modules,calls=self.backend()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(self.output) as model:
                with self.assertRaisesRegex(ExecutionPlanError,'Logical memory estimate'):
                    model.predict((Tensor(1),),output_device='furiosa')
                self.assertFalse(any(c[0]=='run' for c in calls))
                self.assertFalse(model.failed)
    def test_package_metadata_swap_after_construction_stops_before_loading(self):
        self.export();model=ExecutionFXB(self.output)
        with zipfile.ZipFile(self.output) as z:files={n:z.read(n) for n in z.namelist()}
        meta=json.loads(files[PRODUCER_MEMBER]);meta['execution_plan']['source_metadata']['architecture']='ChangedFixture'
        files[PRODUCER_MEMBER]=json.dumps(meta).encode()
        with zipfile.ZipFile(self.output,'w') as z:
            for n,v in files.items():z.writestr(n,v)
        modules,calls=self.backend()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'):
            with self.assertRaisesRegex(ExecutionPlanError,'hash changed'):
                with model:pass
        self.assertFalse(any(c[0]=='load' for c in calls))
    def test_repeated_state_dict_target_uploads_once_and_gpu_outputs_are_owned(self):
        spec={'shape':[1,2],'stride':[2,1],'dtype':'float32','kind':'parameter','target':'weight'}
        self.plan['tensors'].update(p1=spec,p2=dict(spec));self.plan['entries']['forward']['outputs']=['y','p1','p2']
        self.export();modules,calls=self.backend();uploads=[]
        class Counted(Tensor):
            def to(self,device):uploads.append(device);return super().to(device)
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(self.output,weights={'weight':Counted(3)}) as model:
                self.assertEqual(uploads,['furiosa:0']);self.assertIs(model.resident['p1'],model.resident['p2'])
                out=model.predict((Tensor(1),),output_device='furiosa')
                self.assertIsNot(out[1],model.resident['p1']);out[1].value=100
                self.assertEqual(model.resident['p1'].value,3)
    def test_abi_failure_happens_before_device_allocation(self):
        self.export();modules,calls=self.backend(bad_abi=True)
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'):
            with self.assertRaisesRegex(ExecutionPlanError,'ABI'):
                with ExecutionFXB(self.output):pass
        self.assertFalse(any(c[0]=='load' for c in calls))
    def test_kernel_failure_preserves_committed_state_and_invalidates_session(self):
        self.export();modules,calls=self.backend(fail_step=True)
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(self.output) as model:
                with self.assertRaisesRegex(RuntimeError,'kernel failure'):model.predict((Tensor(1),))
                self.assertEqual(model.states['state'].value,0)
                with self.assertRaisesRegex(ExecutionPlanError,'healthy'):model.predict((Tensor(1),))
    def test_cleanup_sync_error_preserves_the_primary_kernel_failure(self):
        self.export();modules,calls=self.backend(fail_step=True)
        def synchronize():
            if any(c[0]=='run' and c[1]=='double' for c in calls):raise RuntimeError('cleanup synchronization failed')
        modules['furiosa.torch'].current_stream=lambda device:types.SimpleNamespace(synchronize=synchronize)
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'kernel failure') as failure:
                with ExecutionFXB(self.output) as model:model.predict((Tensor(1),))
            self.assertIn('cleanup synchronization failed',str(failure.exception.__cause__))
            self.assertIsNone(model.stack)

if __name__=='__main__':unittest.main()
