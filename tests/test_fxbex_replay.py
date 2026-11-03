import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.export._trace
from fxbex.replay import capture, replay, ReplayError


class Runnable:
    def serialize(self):
        return b'runnable'

    @staticmethod
    def deserialize(data):
        assert data == b'runnable'
        return Runnable()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = 0
        self.compiler = ModuleType('furiosa.torch.compiler')
        self.compiler.compile = self.compile
        self.original = self.compiler.compile
        self.native = SimpleNamespace(compile=self.compile, compile_raw_dfg=self.compile)
        modules = {'furiosa.torch.compiler': self.compiler,
                   'furiosa.torch': SimpleNamespace(ir=SimpleNamespace(Runnable=Runnable),
                       config=SimpleNamespace(get_compiler_config=lambda: None)),
                   'furiosa.torch._C': SimpleNamespace(compiler=self.native)}
        self.patcher = patch.dict('sys.modules', modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1
        self.ep = torch.export.export(Model(), (torch.zeros(2, 3),))

    def compile(self, *args, **kwargs):
        self.calls += 1
        return Runnable()

    def record(self):
        with capture(self.root) as stats:
            self.compiler.compile(self.ep)
        self.assertEqual(stats.compile_calls, 1)
        self.assertIs(self.compiler.compile, self.original)

    def test_capture_replay_and_native_guard(self):
        self.record()
        with replay(self.root) as stats:
            self.assertIsInstance(self.compiler.compile(self.ep, task_name='different-label'), Runnable)
            with self.assertRaisesRegex(ReplayError, 'Native compiler'):
                self.native.compile()
        self.assertEqual(self.calls, 1)
        self.assertEqual(stats.replay_hits, 1)
        self.assertEqual(stats.compile_calls, 0)
        self.assertIs(self.compiler.compile, self.original)

    def test_miss_restores_without_compile(self):
        self.record()
        with self.assertRaisesRegex(ReplayError, 'No recorded'):
            with replay(self.root):
                self.ep.constants['different'] = 1
                self.compiler.compile(self.ep)
        self.assertEqual(self.calls, 1)
        self.assertIs(self.compiler.compile, self.original)

    def test_shape_and_constant_misses(self):
        self.record()
        self.ep.constants['constant'] = torch.tensor([2])
        with replay(self.root):
            with self.assertRaisesRegex(ReplayError, 'No recorded'):
                self.compiler.compile(self.ep)
        self.ep.constants.clear()
        next(n for n in self.ep.graph.nodes if n.op == 'placeholder').meta['val'] = torch.zeros(3, 3)
        with replay(self.root):
            with self.assertRaisesRegex(ReplayError, 'No recorded'):
                self.compiler.compile(self.ep)

    def test_unknown_and_large_constants_rejected_before_compile(self):
        for value in (object(), torch.zeros(300000)):
            self.ep.constants['constant'] = value
            with self.assertRaises(ReplayError):
                with capture(self.root):
                    self.compiler.compile(self.ep)
        self.assertEqual(self.calls, 0)

    def test_config_constant_and_stride_are_in_key(self):
        class Config:
            __module__ = 'furiosa.torch.compiler'
            def __init__(self, text):
                self.text = text
            def __str__(self):
                return self.text
        self.ep.constants['constant'] = torch.tensor([1])
        import sys
        sys.modules['furiosa.torch'].config.get_compiler_config = lambda: Config('a')
        with capture(self.root):
            self.compiler.compile(self.ep)
        with replay(self.root):
            self.compiler.compile(self.ep, Config('a'))
            with self.assertRaisesRegex(ReplayError, 'No recorded'):
                self.compiler.compile(self.ep, Config('b'))
            self.ep.constants['constant'] = torch.tensor([2])
            with self.assertRaisesRegex(ReplayError, 'No recorded'):
                self.compiler.compile(self.ep, Config('a'))
            self.ep.constants['constant'] = torch.tensor([1])
            placeholder = next(n for n in self.ep.graph.nodes if n.op == 'placeholder')
            placeholder.meta['val'] = torch.zeros(3, 2).t()
            with self.assertRaisesRegex(ReplayError, 'No recorded'):
                self.compiler.compile(self.ep, Config('a'))
        self.assertEqual(self.calls, 1)

    def test_corruption_and_compatibility(self):
        self.record()
        path = next((self.root / 'edfs').glob('*'))
        path.write_bytes(b'bad')
        with replay(self.root):
            with self.assertRaisesRegex(ReplayError, 'checksum'):
                self.compiler.compile(self.ep)
        plan = self.root / 'execution_plan.json'
        data = json.loads(plan.read_text())
        data['compatibility']['torch'] = 'different'
        plan.write_text(json.dumps(data))
        with self.assertRaisesRegex(ReplayError, 'compatibility'):
            with replay(self.root):
                pass

    def test_duplicate_capture_and_overwrite(self):
        with capture(self.root):
            self.compiler.compile(self.ep)
            self.compiler.compile(self.ep)
        plan = json.loads((self.root / 'execution_plan.json').read_text())
        self.assertEqual(len(plan['steps']), 1)
        self.assertEqual(self.calls, 2)
        with self.assertRaisesRegex(ReplayError, 'overwrite'):
            with capture(self.root):
                pass


if __name__ == '__main__':
    unittest.main()
