import unittest
from unittest.mock import patch
from fxbex.runtime import package_compatibility
class Compatibility(unittest.TestCase):
 def test_transformers_minor_is_warning(self):
  with patch('fxbex.runtime.environment',return_value={'transformers':'5.22.1'}):
   self.assertEqual(len(package_compatibility({'compatibility':{'packages':{'transformers':'5.20.2'}}})),1)
 def test_abi_missing_and_major_remain_errors(self):
  for name,expected,current in [('torch','2.10','2.11'),('furiosa-torch','2026.4','2026.5'),('transformers','5.20',None),('transformers','5.20','6.0')]:
   with patch('fxbex.runtime.environment',return_value={name:current}):
    with self.assertRaises(ValueError):package_compatibility({'compatibility':{'packages':{name:expected}}})

 def test_observed_qwen_crash_combination_is_blocked(self):
  versions={'furiosa-torch':'2026.4.0','torch':'2.10.0+cpu','transformers':'5.22.1'}
  with patch('fxbex.runtime.environment',return_value=versions),patch('sys.version_info',(3,11,0)):
   with self.assertRaisesRegex(ValueError,'Known unstable'):
    package_compatibility({'model':{'name':'RedHatAI/Qwen3.5-9B-FP8-dynamic'},'compatibility':{'packages':{}}})
