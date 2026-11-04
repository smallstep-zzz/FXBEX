import unittest
from fxbex.provenance import source_metadata
class Provenance(unittest.TestCase):
 def test_explicit_hf_id_and_unknown(self):
  self.assertEqual(source_metadata('stabilityai/TripoSR')['source_url'],'https://huggingface.co/stabilityai/TripoSR')
  self.assertEqual(source_metadata()['status'],'unknown')
  self.assertEqual(source_metadata(existing={'source_url':'https://huggingface.co/a/b','revision':'a'*40})['revision'],'a'*40)
 def test_credentials_not_embedded(self):
  for value in ['http://example.com/model','https://token@example.com/model','https://example.com/?token=x']:
   with self.assertRaises(ValueError):source_metadata(value)
