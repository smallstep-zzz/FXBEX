import unittest
from unittest.mock import patch
import torch
from fxbex.torch_api import CompiledModule,restore

class FakeAdapter:
 config={'inputs':{'x':{}},'output_tree':{'tuple':['y',{'constant':None}]}}
 def predict_tensors(self,*a,**k):pass
class FakeSession:
 def __init__(self,*a):self.adapter=FakeAdapter();self.closed=False
 def __enter__(self):return self
 def __exit__(self,*a):self.closed=True
 def predict_tensors(self,values,**kwargs):return {'outputs':{'y':values['x']+1}}

class TorchApiTests(unittest.TestCase):
 def test_tensor_output_structure_and_close(self):
  with patch('fxbex.torch_api.Session',FakeSession):
   with CompiledModule('dummy') as model:
    result=model(torch.tensor([2],dtype=torch.int32))
    self.assertEqual(result[0].dtype,torch.int32)
    self.assertEqual(result[0].item(),3);self.assertIsNone(result[1])
   self.assertTrue(model.session.closed)
   with self.assertRaises(RuntimeError):model(torch.tensor([2]))
 def test_reject_unsupported_mutation_and_arguments(self):
  with patch('fxbex.torch_api.Session',FakeSession):
   with CompiledModule('dummy') as model:
    self.assertIs(model.eval(),model)
    for operation in (lambda:model.train(),lambda:model.to('cpu'),lambda:model.state_dict()):
     with self.assertRaises(NotImplementedError):operation()
    with self.assertRaises(TypeError):model(torch.ones(1),x=torch.ones(1))
    with self.assertRaises(TypeError):model(extra=torch.ones(1))
 def test_library_output_is_allowlisted(self):
  with self.assertRaises(NotImplementedError):restore({'library_output':{'library':'custom','class':'Arbitrary','fields':{}}},{})

if __name__=='__main__':unittest.main()
