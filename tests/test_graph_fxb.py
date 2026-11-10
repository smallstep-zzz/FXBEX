"""Offline contract tests: fake bytes/backends never count as SDK evidence."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import zipfile

from fxbex.graph_fxb import (ExistingNativeContract, GraphFXB, GraphFXBError,
                             export_graph, read_graph_plan)
from fxbex.native_fxb import PRODUCER_MEMBER


class Tensor:
    def __init__(self, name, *, shape=(1, 2), device='cpu'):
        self.name, self.shape = name, shape
        self.dtype, self.requires_grad = 'torch.float32', False
        self.device = types.SimpleNamespace(type=device.split(':')[0])

    def stride(self): return (2, 1)
    def detach(self): return self
    def to(self, device): return Tensor(self.name, shape=self.shape, device=device)
    def cpu(self): return self.to('cpu')


class GraphFXBTests(unittest.TestCase):
    def test_legacy_writer_cannot_drop_value_constraints(self):
        with self.assertRaisesRegex(GraphFXBError,'canonical execution-plan'):
            export_graph({'input_constraints':[{'kind':'integer_range'}]},'unused.fxb',model_metadata={})
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.edf = self.root / 'graph.edf'
        self.edf.write_bytes(b'fake-edf-only-for-offline-tests')
        self.output = self.root / 'model.fxb'
        self.metadata = {'architecture': 'OfflineFixture', 'source_revision': 'fixture'}
        spec = {'shape': [1, 2], 'stride': [2, 1], 'dtype': 'float32'}
        self.result = {'edf_path': str(self.edf),
                       'edf_sha256': hashlib.sha256(self.edf.read_bytes()).hexdigest(),
                       'inputs': [dict(spec, name='p_weight', kind='parameter', target='weight'),
                                  dict(spec, name='x', kind='user_input', user_index=0)],
                       'outputs': [dict(spec, name='result')],
                       'abi': {'inputs': ['weight ABI', 'user ABI'], 'outputs': ['output ABI'],
                               'logical_to_physical_metadata': 'unknown'},
                       'compatibility': {'torch': 'test', 'furiosa-torch': 'test', 'furiosa-tcc': 'test'},
                       'expected_num_fusion': 8,
                       'validation': {'external_weight_variation': 'not_tested'},
                       'output_tree_spec': 'fixture-tree', 'target_npu': 'renegade'}
        self.manifest = {'format_version': 2, 'uuid': 'offline-fixture',
                         'created_at': '2026-11-08T00:00:00Z', 'versions': {},
                         'model': {'architecture': 'OfflineFixture'}, 'parallel_config': {},
                         'kernels': {'fixture': [{'bucket': {'tokenwise': {'input_size': 1}},
                                                 'file': 'edfs/fixture.edf'}]}}
        self.provider = ExistingNativeContract(
            self.manifest, self.metadata, 'fixture', {'tokenwise': {'input_size': 1}},
            'edfs/fixture.edf', {'kind': 'explicit-test-provider'},
            device_config={'fusion': 8, 'mesh': 'single'})

    def export(self):
        return export_graph(self.result, self.output, model_metadata=self.metadata,
                            contract_provider=self.provider, verify_official=False)

    def mutate(self, callback):
        with zipfile.ZipFile(self.output) as archive:
            files = {x: archive.read(x) for x in archive.namelist()}
        callback(files)
        with zipfile.ZipFile(self.output, 'w') as archive:
            for name, data in files.items(): archive.writestr(name, data)

    def test_automatic_assembly_and_ordered_plan(self):
        report = self.export()
        plan = read_graph_plan(self.output)
        self.assertEqual([x['name'] for x in plan['inputs']], ['p_weight', 'x'])
        self.assertEqual(plan['outputs'], self.result['outputs'])
        self.assertEqual(plan['output_tree_spec'], 'fixture-tree')
        self.assertEqual(report['gates']['graph_plan'], 'passed')
        self.assertEqual(report['gates']['npu_execution'], 'not_run')
        self.assertEqual(plan['abi']['logical_to_physical_metadata'], 'unknown')

    def test_default_provider_rejects_unrecorded_graph_creates_nothing(self):
        with self.assertRaisesRegex(GraphFXBError, 'STATIC_GRAPH_CONTRACT'):
            export_graph({}, self.output, model_metadata=self.metadata)
        self.assertFalse(self.output.exists())

    def test_provider_cannot_select_another_model(self):
        with self.assertRaisesRegex(GraphFXBError, 'different model metadata'):
            export_graph(self.result, self.output, model_metadata={'architecture': 'Other'},
                         contract_provider=self.provider, verify_official=False)
        self.assertFalse(self.output.exists())

    def test_provider_architecture_mismatch_refused(self):
        self.manifest['model']['architecture'] = 'UnrelatedArchitecture'
        with self.assertRaisesRegex(GraphFXBError, 'architecture'):
            self.export()

    def test_changed_edf_or_incomplete_binding_refused(self):
        self.edf.write_bytes(b'changed')
        with self.assertRaisesRegex(GraphFXBError, 'checksum'):
            self.export()
        self.result['edf_sha256'] = hashlib.sha256(b'changed').hexdigest()
        self.result['inputs'][1]['user_index'] = 3
        with self.assertRaisesRegex(GraphFXBError, 'indices'):
            self.export()

    def test_plan_tampering_rejected(self):
        self.export()
        def mutate(files):
            producer = json.loads(files[PRODUCER_MEMBER])
            producer['graph_plan']['inputs'].reverse()
            files[PRODUCER_MEMBER] = json.dumps(producer).encode()
        self.mutate(mutate)
        with self.assertRaisesRegex(GraphFXBError, 'changed graph execution plan'):
            read_graph_plan(self.output)

    def test_edf_hash_verified_independently_of_zip_crc(self):
        self.export()
        self.mutate(lambda files: files.update({'edfs/fixture.edf': b'changed with valid ZIP CRC'}))
        with self.assertRaisesRegex(GraphFXBError, 'EDF checksum'):
            read_graph_plan(self.output)

    def backend(self, *, mismatch=False):
        torch = types.ModuleType('torch')
        torch.Tensor, torch.no_grad = Tensor, nullcontext
        torch.device = lambda x: x
        sdk = types.ModuleType('furiosa.torch')
        calls = []
        state={'fusion':1}
        def preprocess(raw):
            calls.append(('preprocess',state['fusion']))
            return types.SimpleNamespace(input_dram_shapes=['BAD','user ABI'] if mismatch else ['weight ABI','user ABI'],
                output_dram_shapes=['output ABI'],expected_num_fusion=state['fusion'])
        sdk.ir=types.SimpleNamespace(Runnable=types.SimpleNamespace(from_serialized_edf=preprocess))
        class Program:
            def to(self, device): calls.append(('load', device)); return self
            def __call__(self, *args):
                calls.append(('call', [x.name for x in args]))
                return [Tensor('output', device='furiosa')]
        sdk.Runnable = lambda runnable: Program()
        def configure(value):state['fusion']=value;calls.append(('fusion',value))
        sdk.set_fusion = configure
        sdk.set_mesh_kind = lambda value: None
        sdk.MeshKind = types.SimpleNamespace(Single=lambda: 'single')
        sdk.current_stream = lambda device: types.SimpleNamespace(synchronize=lambda: calls.append(('sync',)))
        config = types.ModuleType('furiosa.torch.config')
        config.set_cpu_fallback = lambda value: None
        config.get_cpu_fallback = lambda: False
        package = types.ModuleType('furiosa')
        package.torch = sdk
        return {'torch': torch, 'furiosa': package, 'furiosa.torch': sdk,
                'furiosa.torch.config': config}, calls

    def test_runtime_uses_saved_manifest_reference_and_binding_order(self):
        self.export()
        modules, calls = self.backend()
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='test'), \
                patch('fxbex.settings.npu_lock', return_value=nullcontext()):
            with GraphFXB(self.output, weights={'weight': Tensor('weight')}) as model:
                output = model.predict({'x': Tensor('input')})
                self.assertEqual(output[0].device.type, 'cpu')
                with self.assertRaisesRegex(GraphFXBError, 'names differ'):
                    model.predict({'wrong': Tensor('input')})
            self.assertIn(('call', ['weight', 'input']), calls)
            self.assertLess(calls.index(('fusion',8)),calls.index(('preprocess',8)))
            self.assertIsNone(model.program)

    def test_cpu_output_cannot_satisfy_npu_contract(self):
        self.export();modules,calls=self.backend()
        original=modules['furiosa.torch'].Runnable
        def program(value):
            result=original(value)
            result.__class__.__call__=lambda self,*args:[Tensor('wrong-cpu-output',device='cpu')]
            return result
        modules['furiosa.torch'].Runnable=program
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with GraphFXB(self.output,weights={'weight':Tensor('weight')}) as model:
                with self.assertRaisesRegex(GraphFXBError,'output metadata'):
                    model.predict((Tensor('x'),),output_device='furiosa')
                self.assertTrue(model.failed)

    def test_cpu_weight_upload_stops_before_execution(self):
        self.export();modules,calls=self.backend()
        class CPUWeight(Tensor):
            def to(self,device):return Tensor(self.name,device='cpu')
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            model=GraphFXB(self.output,weights={'weight':CPUWeight('weight')})
            with self.assertRaisesRegex(GraphFXBError,'Weight upload'):
                model.__enter__()
            self.assertIsNone(model.program);self.assertIsNone(model.stack)
            self.assertFalse(any(c[0]=='call' for c in calls))

    def test_cpu_input_upload_stops_before_execution(self):
        self.export();modules,calls=self.backend()
        class CPUInput(Tensor):
            def to(self,device):return Tensor(self.name,device='cpu')
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with GraphFXB(self.output,weights={'weight':Tensor('weight')}) as model:
                with self.assertRaisesRegex(GraphFXBError,'Input upload'):
                    model.predict((CPUInput('x'),))
                self.assertFalse(any(c[0]=='call' for c in calls))

    def test_primary_call_error_survives_cleanup_sync_error(self):
        self.export();modules,calls=self.backend()
        primary=RuntimeError('original kernel error');cleanup=RuntimeError('cleanup failed')
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            model=GraphFXB(self.output,weights={'weight':Tensor('weight')})
            with self.assertRaises(RuntimeError) as caught:
                with model:
                    modules['furiosa.torch'].current_stream=lambda device:types.SimpleNamespace(synchronize=lambda:(_ for _ in ()).throw(cleanup))
                    with patch.object(type(model.program),'__call__',side_effect=primary):
                        model.predict((Tensor('x'),))
            self.assertIs(caught.exception,primary);self.assertIs(primary.__cause__,cleanup)
            self.assertIsNone(model.program);self.assertIsNone(model.stack);self.assertEqual(model.resident_weights,{})

    def test_primary_load_error_survives_cleanup_sync_error(self):
        self.export();modules,calls=self.backend()
        primary=RuntimeError('upload failed');cleanup=RuntimeError('cleanup failed')
        class BadWeight(Tensor):
            def to(self,device):raise primary
        modules['furiosa.torch'].current_stream=lambda device:types.SimpleNamespace(synchronize=lambda:(_ for _ in ()).throw(cleanup))
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            model=GraphFXB(self.output,weights={'weight':BadWeight('weight')})
            with self.assertRaises(RuntimeError) as caught:model.__enter__()
            self.assertIs(caught.exception,primary);self.assertIs(primary.__cause__,cleanup)
            self.assertIsNone(model.program);self.assertIsNone(model.stack)

    def test_runtime_abi_mismatch_stops_before_device_loading(self):
        self.export()
        modules, calls = self.backend(mismatch=True)
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='test'), patch('fxbex.settings.npu_lock', return_value=nullcontext()):
            with self.assertRaisesRegex(GraphFXBError, 'ABI differs'):
                with GraphFXB(self.output, weights={'weight': Tensor('weight')}): pass
        self.assertFalse(any(call[0]=='load' for call in calls))

    def test_unknown_device_configuration_fails_before_hardware(self):
        self.provider.device_config = None
        self.export()
        modules, calls = self.backend()
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='test'):
            with self.assertRaisesRegex(GraphFXBError, 'RUNTIME_BRIDGE_GAP'):
                with GraphFXB(self.output, weights={'weight': Tensor('weight')}): pass
        self.assertFalse(calls)

    def test_legacy_qualification_is_rejected_explicitly(self):
        self.export()
        with self.assertRaisesRegex(GraphFXBError,'native execution-plan'):
            GraphFXB(self.output,weights={'weight':Tensor('weight')},qualification='receipt.json')

    def test_busy_lease_precedes_configuration_and_deserialization(self):
        self.export();modules,calls=self.backend()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',side_effect=RuntimeError('busy')):
            with self.assertRaisesRegex(RuntimeError,'busy'):
                with GraphFXB(self.output,weights={'weight':Tensor('weight')}):pass
        self.assertEqual(calls,[])

    def test_partial_program_move_is_owned_and_completed_before_release(self):
        self.export();modules,calls=self.backend()
        model=GraphFXB(self.output,weights={'weight':Tensor('weight')})
        class FailedMove:
            def to(inner,device):
                self.assertIs(model.program,inner)
                raise RuntimeError('partial move failed')
        modules['furiosa.torch'].Runnable=lambda ir:FailedMove()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'partial move failed'):model.__enter__()
        self.assertIn(('sync',),calls)
        self.assertIsNone(model.program);self.assertIsNone(model.stack)

    def test_load_completes_transfers_before_returning(self):
        self.export();modules,calls=self.backend()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            model=GraphFXB(self.output,weights={'weight':Tensor('weight')}).__enter__()
            self.assertEqual(calls[-1],('sync',))
            model.close()

    def test_predict_and_close_are_serialized(self):
        self.export();modules,calls=self.backend()
        entered=threading.Event();release=threading.Event();closing=threading.Event();closed=threading.Event();errors=[]
        class BlockingProgram:
            def to(self,device):return self
            def __call__(self,*args):
                entered.set()
                if not release.wait(2):raise RuntimeError('test release missing')
                return [Tensor('output',device='furiosa')]
        modules['furiosa.torch'].Runnable=lambda ir:BlockingProgram()
        def invoke(call):
            try:call()
            except BaseException as error:errors.append(error)
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            model=GraphFXB(self.output,weights={'weight':Tensor('weight')}).__enter__()
            caller=threading.Thread(target=invoke,args=(lambda:model.predict((Tensor('x'),)),))
            def close():closing.set();model.close();closed.set()
            closer=threading.Thread(target=invoke,args=(close,))
            caller.start()
            try:
                self.assertTrue(entered.wait(2))
                acquired=model._lock.acquire(blocking=False)
                if acquired:model._lock.release()
                self.assertFalse(acquired,'predict must own the session lock')
                closer.start();self.assertTrue(closing.wait(2))
                self.assertFalse(closed.is_set());self.assertIsNotNone(model.program)
            finally:
                release.set();caller.join(2)
                if closer.ident is not None:closer.join(2)
            self.assertFalse(caller.is_alive() or closer.is_alive())
            self.assertEqual(errors,[]);self.assertTrue(closed.is_set())

    def test_legacy_rejects_unknown_completion_on_same_sdk(self):
        self.export();modules,calls=self.backend()
        stream=types.SimpleNamespace(synchronize=lambda:None)
        modules['furiosa.torch'].current_stream=lambda device:stream
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            model=GraphFXB(self.output,weights={'weight':Tensor('weight')}).__enter__()
            stream.synchronize=lambda:(_ for _ in ()).throw(RuntimeError('completion failed'))
            with self.assertRaisesRegex(RuntimeError,'completion failed'):model.close()
            self.assertFalse(model.completion_known)
            calls.clear();stream.synchronize=lambda:None
            with self.assertRaisesRegex(GraphFXBError,'NATIVE_COMPLETION_UNKNOWN'):
                with GraphFXB(self.output,weights={'weight':Tensor('weight')}):pass
            self.assertEqual(calls,[])


if __name__ == '__main__':
    unittest.main()
