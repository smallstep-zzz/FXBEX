import unittest
import time
from unittest.mock import patch,MagicMock
from fxbex import LLM

class LLMTest(unittest.TestCase):
    def setUp(self):
        self.inspect=patch('fxbex.model.inspect',return_value={'model':{'name':'test/model'}})
        self.inspect.start();self.addCleanup(self.inspect.stop)
        self.service=patch('fxbex.model.PackageService');self.mock=self.service.start();self.addCleanup(self.service.stop)
        self.mock.return_value.chat.return_value={'text':'4','metrics':{}}
    def test_alias_and_lifecycle(self):
        for kw in ['fxb','fxbex']:
            with LLM(**{kw:'model.fxbex'}) as model:
                self.assertEqual(model.generate('2+2')['text'],'4')
            self.mock.return_value.unload.assert_called()
    def test_package_positional(self):
        with LLM('model.fxbex') as model:self.assertEqual(model.model,'test/model')
        self.mock.assert_called_with('model.fxbex')
    def test_exclusive_arguments(self):
        for kwargs in [{},{'fxb':'x','fxbex':'y'}]:
            with self.assertRaises(ValueError):LLM(**kwargs)
    def test_history_failure_reset(self):
        model=LLM(fxbex='x');model.converse('hi')
        self.assertEqual(len(model.history),2)
        self.mock.return_value.chat.side_effect=RuntimeError('failed')
        with self.assertRaises(RuntimeError):model.converse('next')
        self.assertEqual(len(model.history),2)
        model.reset();self.assertEqual(model.history,[]);model.close()
    def test_converse_uses_one_deadline_through_nested_chat(self):
        model=LLM('x.fxbex');deadline=time.monotonic()+10
        with patch('fxbex.model.request_deadline',return_value=deadline) as policy:
            model.converse('hi')
        policy.assert_called_once_with('request')
        self.assertEqual(self.mock.return_value.chat.call_args.kwargs['deadline'],deadline)
        model.close()
