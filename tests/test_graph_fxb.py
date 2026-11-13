"""Offline contract tests: fake bytes/backends never count as SDK evidence."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
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
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', side_effect=lambda name: 'new' if name=='furiosa-tcc' else 'test'), \
                patch('fxbex.settings.npu_lock', return_value=nullcontext()), self.assertWarnsRegex(RuntimeWarning, 'SDK mismatch furiosa-tcc'):
            with GraphFXB(self.output, weights={'weight': Tensor('weight')}) as model:
                output = model.predict({'x': Tensor('input')})
                self.assertEqual(output[0].device.type, 'cpu')
                with self.assertRaisesRegex(GraphFXBError, 'names differ'):
                    model.predict({'wrong': Tensor('input')})
            self.assertIn(('call', ['weight', 'input']), calls)
            self.assertLess(calls.index(('fusion',8)),calls.index(('preprocess',8)))
            self.assertIsNone(model.program)

    def test_runtime_abi_mismatch_stops_before_device_loading(self):
        self.export()
        modules, calls = self.backend(mismatch=True)
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='test'):
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


if __name__ == '__main__':
    unittest.main()
