import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from fxbex.cli import _runtime_call
class Output(unittest.TestCase):
 def test_default_logs_and_verbose_prints(self):
  def model():
   print('NPU EXECUTE START');print('SDK diagnostic',file=sys.stderr);return {'text':'Hello'}
  with tempfile.TemporaryDirectory() as d:
   log=Path(d)/'runtime.log';out=io.StringIO();err=io.StringIO()
   with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):r=_runtime_call(model,log)
   self.assertEqual(r['text'],'Hello');self.assertEqual(out.getvalue(),'');self.assertEqual(err.getvalue(),'');self.assertIn('NPU EXECUTE START',log.read_text())
   with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):_runtime_call(model,log,True)
   self.assertIn('NPU EXECUTE START',out.getvalue());self.assertIn('SDK diagnostic',err.getvalue())
 def test_errors_propagate(self):
  with tempfile.TemporaryDirectory() as d:
   def fail():raise ValueError('inference failed')
   with self.assertRaisesRegex(ValueError,'inference failed'):_runtime_call(fail,Path(d)/'log')
