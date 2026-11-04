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
