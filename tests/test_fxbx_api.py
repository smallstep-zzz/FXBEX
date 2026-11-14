from pathlib import Path
import runpy,sys,types,unittest
from unittest.mock import patch
import fxbx,fxbex
class PublicAPI(unittest.TestCase):
 def test_exports_share_the_artifact_engine_without_copying_implementation(self):
  for name in ('LLM','Model','YOLO','ImageGeneration','hug','as_torch','GraphFXB','SafeTensorWeights'):
   self.assertIs(getattr(fxbx,name),getattr(fxbex,name))
  self.assertEqual(fxbx.__version__,'0.9.0')
 def test_old_adapter_can_import_package_without_access_to_a_writer(self):
  from fxbex.package import PackageError, build
  with self.assertRaisesRegex(PackageError,'cannot create'):
   build('compiled_model','model.fxbx')
  self.assertFalse(hasattr(fxbx,'build_model'))
  self.assertFalse(hasattr(fxbx,'build_program'))
 def test_shadowed_legacy_engine_has_actionable_fail_closed_error(self):
  engine=types.ModuleType('fxbex');engine.__file__='/other-sdk/site-packages/fxbex/__init__.py'
  engine.__all__=[];engine.__version__='legacy'
  with patch.dict(sys.modules,{'fxbex':engine}):
   with self.assertRaisesRegex(ImportError,'namespace collision.*non-editably'):
    runpy.run_path(fxbx.__file__,run_name='isolated_fxbx')
 def test_matching_bundled_engine_reexports_without_sdk_discovery(self):
  engine=types.ModuleType('fxbex')
  engine.__file__=str(Path(fxbx.__file__).resolve().parent.parent/'fxbex/__init__.py')
  engine.__all__=['sentinel'];engine.__version__='fixture';engine.sentinel=object()
  before={name for name in sys.modules if name=='furiosa' or name.startswith('furiosa.')}
  with patch.dict(sys.modules,{'fxbex':engine}):
   result=runpy.run_path(fxbx.__file__,run_name='isolated_fxbx')
  self.assertIs(result['sentinel'],engine.sentinel)
  self.assertEqual(result['__version__'],'fixture')
  self.assertEqual(result['__all__'],['sentinel'])
  self.assertEqual({name for name in sys.modules if name=='furiosa' or name.startswith('furiosa.')},before)
