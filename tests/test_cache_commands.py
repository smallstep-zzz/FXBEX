import hashlib,io,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fxbex.cli import main
from fxbex.hub import resolve_package

class CacheCommands(unittest.TestCase):
 def test_cleanup_requires_confirmation_and_preserves_external_files(self):
  with tempfile.TemporaryDirectory() as d,patch('fxbex.settings.cache_root',return_value=Path(d)/'cache'):
   outside=Path(d)/'user.fxbex';outside.write_bytes(b'keep')
   cached=Path(d)/'cache/models/org/model/v1/model.fxbex';cached.parent.mkdir(parents=True);cached.write_bytes(b'cache')
   self.assertEqual(main(['cache','clear']),1);self.assertTrue(cached.exists())
   self.assertEqual(main(['cache','clear','--yes']),0)
   self.assertTrue(outside.exists());self.assertFalse(cached.exists())
 def test_refresh_forces_download_and_verifies_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);download=root/'download';download.write_bytes(b'fresh')
   target=root/'cache/models/org/model/v1/model.fxbex';target.parent.mkdir(parents=True);target.write_bytes(b'old')
   entry={'model':{'name':'fixture','source_repo':'org/model'},'release':{'version':'v1','artifact_url':'https://huggingface.co/org/release/resolve/'+'a'*40+'/model.fxbex','author_sha256':hashlib.sha256(b'fresh').hexdigest()}}
   def get(*args,**kwargs):self.assertTrue(kwargs['force_download']);return str(download)
   with patch('fxbex.settings.cache_root',return_value=root/'cache'),patch('fxbex.hub.urlopen',return_value=io.BytesIO(json.dumps({'models':[entry]}).encode())),patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(hf_hub_download=get)}):
    self.assertEqual(resolve_package('fixture',refresh=True),target.resolve())
    self.assertEqual(target.read_bytes(),b'fresh')
