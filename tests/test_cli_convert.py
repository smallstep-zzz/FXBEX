import contextlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from fxbex.cli import main

class ConversionCLI(unittest.TestCase):
    def test_model_shorthand_delegates_to_official_build(self):
        with patch('fxbex.settings.runtime_python',return_value=__import__('sys').executable),patch('fxbex.model_export.build_model',return_value={'status':'built_not_runtime_validated'}) as build,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['org/model','-o','result.fxb']),0)
        build.assert_called_once_with('org/model','result.fxb',concurrency=4)

    def test_native_inspect_has_no_package_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.fxb'
            with zipfile.ZipFile(path,'w') as z:z.writestr('manifest.json',json.dumps({'format_version':2,'model':{'architecture':'official'},'versions':{}}))
            out=io.StringIO()
            with patch('fxbex.native_fxb.validate_native',return_value={'producer_metadata':None}),patch('fxbex.hub.resolve_package',side_effect=AssertionError('no catalog')),contextlib.redirect_stdout(out):
                self.assertEqual(main(['inspect',str(path)]),0)
            self.assertIsNone(json.loads(out.getvalue())['execution_plan'])
