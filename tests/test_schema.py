import unittest
from unittest.mock import patch
from fxbex.schema import describe,example_inputs
class Schema(unittest.TestCase):
 def test_missing_not_guessed(self):
  with patch('fxbex.schema.inspect',return_value={}):
   with self.assertRaises(ValueError):describe('x')
 def test_example_is_data(self):
  c={'schema_version':1,'method':'chat','inputs':{'messages':{'type':'array'}},'outputs':{'text':{'type':'string'}},'example_input':{'messages':[{'role':'user','content':'Hello'}]}}
  with patch('fxbex.schema.inspect',return_value={'io_contract':c}):self.assertEqual(example_inputs('x'),c['example_input'])
