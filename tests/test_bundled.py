import tempfile
import unittest
from pathlib import Path
import sys
from fxbex.bundled import activate,specification
from fxbex.package import PackageError

class BundleTest(unittest.TestCase):
    def manifest(self):
        return {'runtime':{'bundled_python':{'root':'python','packages':['test_bundle_model'],'entrypoint':'test_bundle_model.adapter:Adapter'}},'files':{'python/test_bundle_model/__init__.py':{},'python/test_bundle_model/adapter.py':{}}}
    def test_isolated_import_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'python/test_bundle_model';p.mkdir(parents=True)
            (p/'__init__.py').write_text('');(p/'adapter.py').write_text('class Adapter: pass')
            with activate(root,self.manifest()) as cls:
                self.assertEqual(cls.__name__,'Adapter')
                with self.assertRaises(PackageError):
                    with activate(root,self.manifest()):pass
            self.assertNotIn('test_bundle_model.adapter',sys.modules)
    def test_reject_missing_source(self):
        m=self.manifest();m['files'].pop('python/test_bundle_model/adapter.py')
        with self.assertRaises(PackageError):specification(m)
