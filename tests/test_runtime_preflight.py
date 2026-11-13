import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.package import PackageError
from fxbex.runtime import Session, RunnableDAG


class RuntimePreflight(unittest.TestCase):
    def test_runnable_fusion_is_checked_before_weight_upload(self):
        from types import ModuleType, SimpleNamespace
        from unittest.mock import Mock
        import sys

        sdk = ModuleType('furiosa.torch')
        sdk.__path__ = []
        package = ModuleType('furiosa')
        package.__path__ = []
        package.torch = sdk
        sdk.ir = SimpleNamespace(Runnable=SimpleNamespace(
            deserialize=Mock(return_value=SimpleNamespace(expected_num_fusion=1))))
        sdk.Runnable = Mock()
        sdk.MeshKind = SimpleNamespace(Single=Mock())
        sdk.set_fusion, sdk.set_mesh_kind = Mock(), Mock()
        config = SimpleNamespace(set_cpu_fallback=Mock(), get_cpu_fallback=Mock(return_value=False))
        torch = SimpleNamespace(load=Mock())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'kernel').write_bytes(b'test input')
            (root/'runtime_config.json').write_text('{}')
            (root/'execution_plan.json').write_text(json.dumps({
                'format': 'runnable-dag-v1',
                'steps': [{'id': 'text', 'artifact': 'kernel'}],
            }))
            with patch('fxbex.memory.analyze', return_value={}), patch.dict(sys.modules, {
                'furiosa': package, 'furiosa.torch': sdk, 'furiosa.torch.config': config, 'torch': torch,
            }):
                with self.assertRaisesRegex(PackageError, 'text fusion mismatch: compiled=1, runtime=8'):
                    RunnableDAG(root)
            torch.load.assert_not_called()
            sdk.Runnable.assert_not_called()

    def test_output_conversion_preserves_large_integers_and_booleans(self):
        from types import SimpleNamespace
        class Tensor:
            def __init__(self,value):self.value=value
            def cpu(self):return self
            def tolist(self):return self.value
            def float(self):raise AssertionError('output must not be coerced to float')
        dag=object.__new__(RunnableDAG)
        dag.torch=SimpleNamespace(no_grad=contextlib.nullcontext)
        dag.current_stream=None
        dag.config={'inputs':{},'outputs':['ids','mask']}
        dag.constants={'ids':Tensor([2**60+1]),'mask':Tensor([True,False])}
        dag.steps=[];dag.memory_plan={}
        result=dag.predict({})
        self.assertEqual(result['outputs']['ids'],[2**60+1])
        self.assertIs(result['outputs']['mask'][0],True)

    def test_malformed_dag_is_rejected_before_device_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data={'manifest.json':{'runtime':{'adapter':'runnable_dag'},'files':{'kernel':{}}},
                  'weights_index.json':{'weights':[]},
                  'runtime_config.json':{'inputs':{},'outputs':['y']},
                  'execution_plan.json':{'format':'runnable-dag-v1','steps':[
                      {'id':'bad','inputs':['unbound'],'outputs':['y'],'artifact':'kernel'}]}}
            for name,value in data.items():(root/name).write_text(json.dumps(value))
            with patch('fxbex.runtime.extract',return_value=contextlib.nullcontext(root)), \
                 patch('fxbex.runtime.check_compatibility'), \
                 patch('fxbex.memory.analyze') as analyze, \
                 patch.object(RunnableDAG,'__init__') as adapter:
                with self.assertRaisesRegex(PackageError,'Unbound'):
                    with Session('unused',report_dir=root/'reports'):pass
                analyze.assert_not_called()
                adapter.assert_not_called()
