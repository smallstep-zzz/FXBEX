import hashlib,io,json,sys,tempfile,unittest,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fxbex.hub import resolve_package
from fxbex.schema import resolve_sample
class Hub(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.old=os.getcwd();os.chdir(self.tmp.name)
  self.addCleanup(self.tmp.cleanup);self.addCleanup(os.chdir,self.old)

 def test_pinned_download_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.fxbex';p.write_bytes(b'package')
   entry={'model':{'name':'Model','source_repo':'org/model'},'release':{'version':'v1','artifact_url':'https://huggingface.co/org/release/resolve/'+'a'*40+'/model.fxbex','author_sha256':hashlib.sha256(b'package').hexdigest()}}
   def response():return io.BytesIO(json.dumps({'models':[entry]}).encode())
   def download(repo,file,revision,**kwargs):
    self.assertEqual(revision,'a'*40);return str(p)
   with patch('fxbex.hub.urlopen',side_effect=lambda *a,**k:response()),patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=download)}):
    self.assertEqual(resolve_package('org/model@v1').read_bytes(),b'package')
    self.assertEqual(resolve_package('./org/model@v1').read_bytes(),b'package')
    self.assertEqual(resolve_package('./org/model').read_bytes(),b'package')
    (Path('org/model/v1')/'model.fxbex').unlink()
    p.write_bytes(b'changed')
    with self.assertRaisesRegex(ValueError,'checksum'):resolve_package('org/model')
 def test_assets_are_explicit_and_confined(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'image.png').write_bytes(b'fixture')
   c={'example_input':{'image':{'$asset':'image.png'},'prompt':'image.png'}}
   r=resolve_sample(c,root,{'files':{'image.png':{}}})
   self.assertEqual(r['image'],str((root/'image.png').resolve()));self.assertEqual(r['prompt'],'image.png')
   with self.assertRaises(ValueError):resolve_sample(c,root,{'files':{}})

 def test_missing_file_paths_do_not_trigger_download(self):
  with patch('fxbex.hub.urlopen') as network:
   for path in ('./missing.fxbex','../org/model','/missing/org/model'):
    with self.assertRaises(FileNotFoundError):resolve_package(path)
   network.assert_not_called()
 def test_local_file_precedes_catalog(self):
  with tempfile.TemporaryDirectory() as d,patch('fxbex.hub.urlopen') as network:
   path=Path(d)/'model';path.write_bytes(b'local')
   self.assertEqual(resolve_package(path),path.resolve());network.assert_not_called()

 def test_version_selection_and_legacy(self):
  base=Path('org/model');base.mkdir(parents=True)
  (base/'model.fxbex').write_bytes(b'legacy')
  for version in ('opt1','opt2'):
   folder=base/version;folder.mkdir();(folder/'model.fxbex').write_bytes(version.encode())
  with patch('fxbex.hub.urlopen') as network:
   self.assertEqual(resolve_package('org/model').read_bytes(),b'legacy')
   self.assertEqual(resolve_package('org/model@opt2').read_bytes(),b'opt2')
   network.assert_not_called()
  (base/'model.fxbex').unlink()
  with self.assertRaisesRegex(ValueError,'Multiple installed versions'):resolve_package('org/model')
  with self.assertRaises(ValueError):resolve_package('org/model@../bad')
