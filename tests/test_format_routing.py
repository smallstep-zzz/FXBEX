"""Native archives must never be treated as damaged FXB_EX packages."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from fxbex import Model, LLM
from fxbex.package import inspect, PackageError
from fxbex.graph_fxb import GraphFXB, GraphFXBError


class FormatRouting(unittest.TestCase):
    def native(self, path):
        manifest={'format_version':2,'uuid':'fixture','created_at':'test','versions':{},
                  'model':{'architecture':'Fixture'},'parallel_config':{},
                  'kernels':{'k':[{'bucket':{},'file':'edfs/kernel.edf'}]}}
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            archive.writestr('edfs/kernel.edf',b'offline-test-not-a-genuine-edf')

    def test_native_named_as_fxbex_gets_loader_guidance_before_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'renamed.fxbex';self.native(path)
            with patch('fxbex.model.PackageService') as service:
                for cls in (Model,LLM):
                    with self.assertRaisesRegex(PackageError,'NATIVE_FXB_REQUIRES_EXPLICIT_LOADER') as caught:
                        cls(path)
                    self.assertIn('GraphFXB',str(caught.exception))
                    self.assertIn('furiosa_llm.LLM',str(caught.exception))
                service.assert_not_called()

    def test_cli_native_inspect_explains_loader_without_traceback(self):
        import contextlib,io
        from fxbex.cli import main
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'native.fxb';self.native(path)
            output,error=io.StringIO(),io.StringIO()
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
                code=main(['inspect',str(path)])
            self.assertEqual(code,1)
            self.assertIn('NATIVE_FXB_REQUIRES_EXPLICIT_LOADER',error.getvalue())
            self.assertNotIn('Traceback',error.getvalue())

    def test_legacy_missing_member_does_not_become_native(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'legacy.fxb'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('manifest.json',json.dumps({'format':'FXB_EX','schema_version':1}))
            with self.assertRaisesRegex(PackageError,'Missing required member: execution_plan.json'):
                inspect(path)

    def test_duplicate_manifest_is_rejected_before_any_routing(self):
        import warnings
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.fxb';self.native(path)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(path,'a') as archive:archive.writestr('manifest.json','{}')
            with self.assertRaisesRegex(PackageError,'Duplicate'):
                inspect(path)

    def test_official_native_needs_official_loader_not_invented_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'official.fxb';self.native(path)
            with self.assertRaisesRegex(GraphFXBError,'NATIVE_FXB_WITHOUT_GRAPH_PLAN'):
                GraphFXB(path)
