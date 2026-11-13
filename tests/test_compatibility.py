import unittest
from unittest.mock import patch
from fxbex.runtime import package_compatibility
class Compatibility(unittest.TestCase):
 def test_transformers_minor_is_warning(self):
  with patch('fxbex.runtime.environment',return_value={'transformers':'5.22.1'}):
   self.assertEqual(len(package_compatibility({'compatibility':{'packages':{'transformers':'5.20.2'}}})),1)
 def test_version_differences_warn_by_default(self):
  for name,expected,current in [('torch','2.10','2.11'),('furiosa-torch','2026.4','2026.5'),('transformers','5.20','6.0')]:
   with self.subTest(name=name), patch('fxbex.runtime.environment',return_value={name:current}):
    self.assertIn('attempting execution',package_compatibility({'compatibility':{'packages':{name:expected}}})[0])

 def test_edf_requires_runtime_but_not_compiler(self):
  from importlib.metadata import PackageNotFoundError
  from fxbex.compatibility import check_edf_versions
  with patch('importlib.metadata.version', side_effect=PackageNotFoundError):
   check_edf_versions({'furiosa-tcc':'2026.4.0'}, ValueError)
   with self.assertRaisesRegex(ValueError,'Missing required runtime package: furiosa-torch'):
    check_edf_versions({'furiosa-torch':'2026.4.0'}, ValueError)

 def test_opt_in_reports_version_mismatch(self):
  manifest={'compatibility':{'packages':{'furiosa-tcc':'2026.4.0'}}}
  with patch('fxbex.runtime.environment',return_value={'furiosa-tcc':'2026.4.1'}):
   notices=package_compatibility(manifest,allow_sdk_mismatch=True)
   self.assertIn('attempting execution',notices[0])

 def test_opt_in_still_reports_all_missing_packages(self):
  manifest={'compatibility':{'packages':{'furiosa-tcc':'2026.4.0','torch':'2.10.0+cpu'}}}
  with patch('fxbex.runtime.environment',return_value={}):
   with self.assertRaisesRegex(ValueError,'Python .*furiosa-tcc==2026.4.0, torch==2.10.0'):
    package_compatibility(manifest,allow_sdk_mismatch=True)

 def test_observed_qwen_crash_combination_is_blocked(self):
  versions={'furiosa-torch':'2026.4.0','torch':'2.10.0+cpu','transformers':'5.22.1'}
  with patch('fxbex.runtime.environment',return_value=versions),patch('sys.version_info',(3,11,0)):
   with self.assertRaisesRegex(ValueError,'Known unstable'):
    package_compatibility({'model':{'name':'RedHatAI/Qwen3.5-9B-FP8-dynamic'},'compatibility':{'packages':{}}},allow_sdk_mismatch=True)
