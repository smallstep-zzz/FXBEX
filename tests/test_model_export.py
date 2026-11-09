import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.model_export import build_model, resume_graph_build


class ModelExportTests(unittest.TestCase):
    def test_custom_graph_does_not_silently_ignore_context_limit(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'model.fxb'
            with self.assertRaisesRegex(ValueError,'STATIC_GRAPH_OPTIONS_UNSUPPORTED'):
                build_model(object(),out,example_inputs=(),max_model_len=262144)
            self.assertFalse(out.exists())
    def test_resume_reuses_retained_edf_contract_without_compiler(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            (work/'result.json').write_text(json.dumps({'source_metadata':{'architecture':'Real'}}))
            (work/'edf-contract.json').write_text(json.dumps({'edf_path':'recorded.edf'}))
            with patch('fxbex.graph_fxb.export_graph',return_value='packaged') as writer, \
                 patch('fxbex.model_export._official_build',side_effect=AssertionError('must not recompile')):
                self.assertEqual(resume_graph_build(work,'out.fxb',contract_provider='trusted'),'packaged')
            writer.assert_called_once_with({'edf_path':'recorded.edf'},'out.fxb',
                model_metadata={'architecture':'Real'},contract_provider='trusted')

    def test_custom_build_consumes_tclmachine_owned_path(self):
        import sys
        from types import ModuleType
        from unittest.mock import Mock
        try:
            import torch
        except ImportError:
            self.skipTest('optional PyTorch not installed')
        retained = {'contract_owner': 'TCLMachine', 'regions': 3}
        module = ModuleType('tplmachine.execution_plan')
        module.export_model = Mock(return_value=retained)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'model.fxb'
            model=torch.nn.Identity();inputs=(torch.ones(1),)
            with patch.dict(sys.modules, {'tplmachine.execution_plan': module}), \
                 patch('torch.export.export', side_effect=AssertionError('consumer must not capture')):
                result = build_model(model, out, example_inputs=inputs, concurrency=4)
            self.assertEqual(result['result'], retained)
            module.export_model.assert_called_once_with(model, inputs, out, source_metadata=None,
                state_inputs=None, state_updates=None, user_output_indices=None, compile_workers=4)
            self.assertFalse((Path(str(out) + '.build') / 'model.edf').exists())

    def test_legacy_provider_rejects_output_filtering_instead_of_ignoring_it(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'no state/output annotations'):
                build_model(object(),Path(d)/'m.fxb',example_inputs=(),
                            contract_provider='trusted',user_output_indices=[0])

    def test_official_failure_keeps_exact_stage_and_no_publish(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'test.fxb'
            with patch('fxbex.model_export._official_build',side_effect=RuntimeError('no model registry entry')):
                with self.assertRaisesRegex(RuntimeError,'registry'):
                    build_model('org/unsupported',out)
            report=json.loads((Path(str(out)+'.build')/'result.json').read_text())
            self.assertEqual(report['status'],'blocked')
            self.assertEqual(report['npu_execution'],'not_run')
            self.assertFalse(out.exists())

    def test_registered_compile_and_writer_automatic_handoff(self):
        import zipfile
        def compile(model,path,**options):
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('manifest.json','{}');z.writestr('edfs/test.edf',b'test-only')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'test.fxb'
            with patch('fxbex.model_export._official_build',side_effect=compile), \
                 patch('fxbex.native_fxb.validate_native'), \
                 patch('fxbex.native_fxb.build_native',return_value={'gates':{'runtime_load':'not_run'}}) as writer:
                result=build_model('org/model',out,concurrency=4)
            self.assertEqual(result['status'],'built_not_runtime_validated')
            self.assertFalse(result['weights_included'])
            self.assertEqual(writer.call_count,1)

    def test_rejects_overwrite_and_unknown_compiler_flags(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'test.fxb';out.write_text('preserve')
            with self.assertRaises(FileExistsError):build_model('org/model',out)
            self.assertEqual(out.read_text(),'preserve')
            with self.assertRaises(ValueError):build_model('org/model',Path(d)/'other.fxb',skip_all_validation=True)
