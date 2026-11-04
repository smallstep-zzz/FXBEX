import hashlib,io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fxbex.hub import resolve_package
from fxbex.schema import resolve_sample
class Hub(unittest.TestCase):
 def test_pinned_download_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.fxbex';p.write_bytes(b'package')
   entry={'model':{'name':'Model','source_repo':'org/model'},'release':{'version':'v1','artifact_url':'https://huggingface.co/org/release/resolve/'+'a'*40+'/model.fxbex','author_sha256':hashlib.sha256(b'package').hexdigest()}}
   def response():return io.BytesIO(json.dumps({'models':[entry]}).encode())
   def download(repo,file,revision):
    self.assertEqual(revision,'a'*40);return str(p)
   with patch('fxbex.hub.urlopen',side_effect=lambda *a,**k:response()),patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=download)}):
    self.assertEqual(resolve_package('org/model@v1'),p)
    p.write_bytes(b'changed')
    with self.assertRaisesRegex(ValueError,'checksum'):resolve_package('org/model')
 def test_assets_are_explicit_and_confined(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'image.png').write_bytes(b'fixture')
   c={'example_input':{'image':{'$asset':'image.png'},'prompt':'image.png'}}
   r=resolve_sample(c,root,{'files':{'image.png':{}}})
   self.assertEqual(r['image'],str((root/'image.png').resolve()));self.assertEqual(r['prompt'],'image.png')
   with self.assertRaises(ValueError):resolve_sample(c,root,{'files':{}})
