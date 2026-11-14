import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from fxbex.cli import main
from fxbex.validation import validate


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'compiled'
        self.source.mkdir()
        configs = {
            'manifest.json': {'model': {'name': 'demo'}, 'runtime': {'adapter': 'runnable_dag'}, 'compatibility': {}},
            'execution_plan.json': {'format': 'runnable-dag-v1', 'steps': [{'id': 'a', 'artifact': 'kernel.edf', 'inputs': ['x'], 'outputs': ['y']}]},
            'weights_index.json': {'weights': []},
            'memory_plan.json': {},
            'runtime_config.json': {'inputs': {'x': {'shape': [1], 'dtype': 'float32'}}, 'outputs': ['y']},
        }
        for name, value in configs.items():
            (self.source / name).write_text(json.dumps(value))
        (self.source / 'kernel.edf').write_bytes(b'fake compiled kernel')
        self.package = self.root / 'model.fxbex'

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(map(str, args)))
        return code, out.getvalue(), err.getvalue()

    def make_package(self):
        members = {p.name: p.read_bytes() for p in self.source.iterdir()}
        manifest = json.loads(members.pop('manifest.json'))
        manifest.update(schema_version=1, format='FXB_EX', standalone=False, files={
            name: {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            for name, data in members.items()
        })
        with zipfile.ZipFile(self.package, 'w') as archive:
            archive.writestr('manifest.json', json.dumps(manifest))
            for name, data in members.items():
                archive.writestr(name, data)

    def test_inspect_and_integrity_only(self):
        self.make_package()
        self.assertEqual(json.loads(self.invoke('inspect', self.package)[1])['model']['name'], 'demo')
        memory = json.loads(self.invoke('inspect', self.package, '--memory')[1])['memory_plan']
        self.assertEqual(memory['kind'], 'dag-logical-memory-v1')
        self.assertEqual(memory['estimate']['unknown_tensors_count'], 1)
        with patch('fxbex.runtime.doctor', side_effect=AssertionError('must not inspect device')):
            code, out, _ = self.invoke('validate', self.package, '--integrity-only')
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(out)['statically_ready'])
        self.assertFalse(json.loads(out)['inference_validated'])

    def test_missing_file_no_traceback(self):
        code, _, err = self.invoke('validate', self.package)
        self.assertEqual(code, 1)
        self.assertIn('fxbx:', err)
        self.assertNotIn('Traceback', err)

    def test_corrupt_package_no_traceback(self):
        self.make_package()
        with zipfile.ZipFile(self.package) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members['kernel.edf'] = b'broken kernel'
        with zipfile.ZipFile(self.package, 'w') as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        code, _, err = self.invoke('validate', self.package, '--integrity-only')
        self.assertEqual(code, 1)
        self.assertIn('mismatch', err)
        self.assertNotIn('Traceback', err)

    def test_help_examples_and_removed_capture(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as raised:
            main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        self.assertNotIn('build', out.getvalue())
        self.assertIn('--port 8000', out.getvalue())
        self.assertNotIn(' build ', out.getvalue())
        self.assertNotIn('capture-qwen', out.getvalue())

    def test_static_validation_no_adapter_instantiation(self):
        self.make_package()
        with patch('fxbex.runtime.check_compatibility'), patch('fxbex.runtime.resolve_adapter') as adapter, patch('fxbex.runtime.doctor', return_value={'device': {'returncode': 0, 'status': 'RNGD ready'}}):
            result = validate(self.package)
        self.assertTrue(result['statically_ready'])
        self.assertFalse(result['inference_validated'])
        adapter.return_value.assert_not_called()

    def test_missing_sdk_and_device_fail_cli(self):
        self.make_package()
        with patch('fxbex.runtime.check_compatibility', side_effect=ValueError('Missing SDK')), patch('fxbex.runtime.resolve_adapter'), patch('fxbex.runtime.doctor', return_value={'device': {'error': 'missing'}}):
            code, out, _ = self.invoke('validate', self.package)
        self.assertEqual(code, 1)
        self.assertIn('Missing SDK', out)
        self.assertFalse(json.loads(out)['statically_ready'])
