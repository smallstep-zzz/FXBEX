import unittest
from unittest.mock import patch,MagicMock
from fxbex import LLM

class LLMTest(unittest.TestCase):
    def setUp(self):
        self.inspect=patch('fxbex.llm.inspect',return_value={'model':{'name':'test/model'}})
        self.inspect.start();self.addCleanup(self.inspect.stop)
        self.service=patch('fxbex.llm.PackageService');self.mock=self.service.start();self.addCleanup(self.service.stop)
        self.mock.return_value.chat.return_value={'text':'4','metrics':{}}
    def test_alias_and_lifecycle(self):
        for kw in ['fxb','fxbex']:
            with LLM('test/model',**{kw:'model.fxbex'}) as model:
                self.assertEqual(model.generate('2+2')['text'],'4')
            self.mock.return_value.unload.assert_called()
    def test_mismatch_before_allocation(self):
        with self.assertRaises(ValueError):LLM('wrong/model',fxb='x')
        self.mock.assert_not_called()
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
