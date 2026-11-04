import unittest
from pathlib import Path
from unittest.mock import patch
from fxbex import Model,YOLO,ImageGeneration

class ModelTest(unittest.TestCase):
 def test_routing_and_timing(self):
  with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
   svc.return_value.predict.return_value={'output':'result.png'}
   with YOLO('detector.fxbex') as m:
    self.assertEqual(m.predict(Path('/tmp/image.png'))['output'],'result.png')
    svc.return_value.predict.assert_called_with({'image':'/tmp/image.png'})
    self.assertGreaterEqual(m.last_timing['wall_seconds'],0)
   with ImageGeneration(fxbex='generator.fxbex') as m:
    m.generate('a bird',seed=42)
    svc.return_value.predict.assert_called_with({'prompt':'a bird','seed':42})
   svc.return_value.unload.assert_called()
 def test_failed_call_is_not_a_successful_latency_sample(self):
  with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
   svc.return_value.predict.side_effect=RuntimeError('failed')
   with Model('x.fxbex') as m:
    with self.assertRaises(RuntimeError):m.predict({})
    self.assertEqual(m.last_timing['status'],'failed')

 def test_ambiguous_arguments(self):
  with self.assertRaises(ValueError):Model('x',fxbex='y')
