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
   self.assertEqual(resolve_package('./org/model').read_bytes(),b'opt2')
   self.assertEqual(resolve_package('org/model@opt2').read_bytes(),b'opt2')
   network.assert_not_called()
  (base/'model.fxbex').unlink()
  self.assertIn(resolve_package('./org/model').read_bytes(),(b'opt1',b'opt2'))
  with self.assertRaises(ValueError):resolve_package('org/model@../bad')

 def test_download_uses_canonical_model_filename(self):
  with tempfile.TemporaryDirectory() as d:
   payload=Path(d)/'download';payload.write_bytes(b'fixture')
   entry={'model':{'name':'Readable alias','source_repo':'org/QwenName'},'release':{'version':'opt2','artifact_url':'https://huggingface.co/org/model/resolve/'+'a'*40+'/model.fxbex','author_sha256':hashlib.sha256(b'fixture').hexdigest()}}
   with patch('fxbex.hub.urlopen',side_effect=lambda *a,**k:io.BytesIO(json.dumps({'models':[entry]}).encode())),patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=lambda *a,**k:str(payload))}):
    expected=(Path('org/QwenName/opt2/QwenName.fxbex')).resolve()
    self.assertEqual(resolve_package('Readable alias'),expected)
    self.assertEqual(resolve_package('Readable alias'),expected)

 def test_latest_file_in_folder_and_short_local_name(self):
  root=Path('org/short');(root/'old').mkdir(parents=True);(root/'new').mkdir()
  old=root/'old/old.fxbex';new=root/'new/new.fxbex';old.write_bytes(b'old');new.write_bytes(b'new')
  os.utime(old,ns=(100,100));os.utime(new,ns=(200,200))
  self.assertEqual(resolve_package(root),new.resolve())
  with patch('fxbex.hub.urlopen',side_effect=OSError('offline')):
   self.assertEqual(resolve_package('short'),new.resolve())
  self.assertEqual(resolve_package('short@old'),old.resolve())

 def test_short_remote_id_uses_registration_order(self):
  from fxbex.hub import catalog_matches
  a={'model':{'name':'Display','source_repo':'org/short'},'release':{'version':'opt9','author_sha256':'a'}}
  b={'model':{'name':'Display','source_repo':'org/short'},'release':{'version':'opt10','author_sha256':'b'}}
  c={'models':[b,a],'release_order':['org/short@opt9#a','org/short@opt10#b']}
  self.assertEqual(catalog_matches(c,'short'),[b])
  self.assertEqual(catalog_matches(c,'short','opt9'),[a])

 def test_catalog_latest_beats_newly_touched_old_local_release(self):
  root=Path('org/short')
  old=root/'old/short.fxbex';new=root/'current/short.fxbex'
  for p,data in [(old,b'old'),(new,b'current')]:
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  os.utime(old,ns=(999999999999,999999999999));os.utime(new,ns=(1,1))
  def entry(version,data):
   return {'model':{'name':'Display','source_repo':'org/short'},'release':{'version':version,'artifact_url':'https://huggingface.co/org/release/resolve/'+'a'*40+'/model.fxbex','author_sha256':hashlib.sha256(data).hexdigest()}}
  a,b=entry('old',b'old'),entry('current',b'current')
  catalog={'models':[a,b],'release_order':['org/short@old#'+a['release']['author_sha256'],'org/short@current#'+b['release']['author_sha256']]}
  with patch('fxbex.hub.urlopen',side_effect=lambda *a,**k:io.BytesIO(json.dumps(catalog).encode())),patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=lambda *a,**k:self.fail('already installed current release must not download'))}):
   self.assertEqual(resolve_package('short'),new.resolve())
   self.assertEqual(resolve_package('org/short'),new.resolve())
   self.assertEqual(resolve_package('./org/short'),old.resolve())
   self.assertEqual(resolve_package(root),old.resolve())
   self.assertEqual(resolve_package('short@old'),old.resolve())

 def test_catalog_checksum_failure_does_not_fall_back_to_older_local(self):
  root=Path('org/short/current');root.mkdir(parents=True);(root/'short.fxbex').write_bytes(b'changed')
  entry={'model':{'name':'Display','source_repo':'org/short'},'release':{'version':'current','artifact_url':'https://huggingface.co/org/release/resolve/'+'a'*40+'/model.fxbex','author_sha256':'b'*64}}
  with patch('fxbex.hub.urlopen',return_value=io.BytesIO(json.dumps({'models':[entry]}).encode())):
   with self.assertRaisesRegex(ValueError,'checksum'):resolve_package('short')
