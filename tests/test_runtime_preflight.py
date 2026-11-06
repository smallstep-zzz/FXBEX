import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.package import PackageError
from fxbex.runtime import Session, RunnableDAG


class RuntimePreflight(unittest.TestCase):
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
