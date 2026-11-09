import unittest
import fxbx,fxbex
class PublicAPI(unittest.TestCase):
 def test_exports_share_the_artifact_engine_without_copying_implementation(self):
  for name in ('LLM','Model','YOLO','ImageGeneration','hug','as_torch','GraphFXB','build_model'):
   self.assertIs(getattr(fxbx,name),getattr(fxbex,name))
  self.assertEqual(fxbx.__version__,'0.8.0')
