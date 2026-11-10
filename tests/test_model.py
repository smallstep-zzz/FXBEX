import unittest
import threading,time
from pathlib import Path
from unittest.mock import patch
from fxbex import Model,YOLO,ImageGeneration

class ModelTest(unittest.TestCase):
 def test_routing_and_timing(self):
  with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
   svc.return_value.predict.return_value={'output':'result.png'}
   with YOLO('detector.fxbex') as m:
    self.assertEqual(m.predict(Path('/tmp/image.png'))['output'],'result.png')
    self.assertEqual(svc.return_value.predict.call_args.args,({'image':'/tmp/image.png'},))
    self.assertGreater(svc.return_value.predict.call_args.kwargs['deadline'],time.monotonic())
    self.assertGreaterEqual(m.last_timing['wall_seconds'],0)
   with ImageGeneration(fxbex='generator.fxbex') as m:
    m.generate('a bird',seed=42)
    self.assertEqual(svc.return_value.predict.call_args.args,({'prompt':'a bird','seed':42},))
   svc.return_value.unload.assert_called()
 def test_failed_call_is_not_a_successful_latency_sample(self):
  with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
   svc.return_value.predict.side_effect=RuntimeError('failed')
   with Model('x.fxbex') as m:
    with self.assertRaises(RuntimeError):m.predict({})
    self.assertEqual(m.last_timing['status'],'failed')

 def test_ambiguous_arguments(self):
  with self.assertRaises(ValueError):Model('x',fxbex='y')

 def test_context_cleanup_preserves_primary_error(self):
  for primary in (RuntimeError('inference failed'),KeyboardInterrupt()):
   with self.subTest(primary=type(primary).__name__):
    with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
     svc.return_value.unload.side_effect=RuntimeError('cleanup failed')
     with self.assertRaises(type(primary)) as failure:
      with Model('x.fxbex'):raise primary
     self.assertIs(failure.exception,primary)
     self.assertEqual(str(failure.exception.__cause__),'cleanup failed')

 def test_cleanup_failure_without_primary_is_visible(self):
  with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
   svc.return_value.unload.side_effect=RuntimeError('cleanup failed')
   with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
    with Model('x.fxbex'):pass

 def test_model_lock_wait_uses_request_deadline(self):
  from fxbex.service import WorkerTimeout
  with patch('fxbex.model.inspect',return_value={'model':{'name':'test'}}),patch('fxbex.model.PackageService') as svc:
   model=Model('x.fxbex');acquired=threading.Event();release=threading.Event()
   def owner():
    with model._lock:acquired.set();release.wait(3)
   thread=threading.Thread(target=owner);thread.start();self.assertTrue(acquired.wait(2))
   try:
    with patch('fxbex.model.request_deadline',return_value=time.monotonic()+.05):
     with self.assertRaises(WorkerTimeout):model.predict({})
    svc.return_value.predict.assert_not_called()
    svc.return_value.cancel.assert_not_called()
   finally:release.set();thread.join(2);model.close()
