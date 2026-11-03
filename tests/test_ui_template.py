import tempfile
import unittest
from pathlib import Path
from fxbex.ui import save_package, sample_code
from fxbex.service import PackageService
from unittest.mock import patch

class TemplateTest(unittest.TestCase):
    def test_copy_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.fxbex';p.write_bytes(b'package')
            target=save_package(p,Path(d)/'fxbex')
            self.assertEqual(Path(target).read_bytes(),b'package')
            self.assertEqual(save_package(target,Path(d)/'fxbex'),target)
            with self.assertRaises(FileExistsError):save_package(p,Path(d)/'fxbex')
            self.assertEqual(Path(target).read_bytes(),b'package')
    def test_code_quotes_filename(self):
        for mode in ['chat','predict']:
            compile(sample_code("fxbex/model'quoted.fxbex",mode),'<sample>','exec')
    def test_predict_requires_load_and_forwards(self):
        s=PackageService('model.fxbex')
        with self.assertRaises(RuntimeError):s.predict({})
        s.loaded=True
        with patch.object(s,'_request',return_value={'ok':True}) as request:
            s.predict({'x':[1]})
            request.assert_called_once_with({'action':'predict','inputs':{'x':[1]}})
