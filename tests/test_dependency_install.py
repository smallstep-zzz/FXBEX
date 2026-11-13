from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fxbex.cli import main
from fxbex.dependencies import MissingPackages, installation_commands, offer_install


class Terminal(io.StringIO):
    def isatty(self):
        return True


class DependencyInstall(unittest.TestCase):
    def test_only_known_packages_and_safe_versions_are_offered(self):
        self.assertEqual(installation_commands({'arbitrary-provider': '1.0'}), [])
        self.assertEqual(installation_commands({'onnx': '1.0; arbitrary'}), [])
        self.assertIn('https://download.pytorch.org/whl/cpu', installation_commands({'torch': '2.10.0+cpu'})[0])
        self.assertIn('huggingface_hub>=0.24,<2', installation_commands({'transformers': '4.35.0'})[0])

    def test_non_interactive_and_decline_do_not_install(self):
        error = MissingPackages({'onnx': '1.20.0'})
        for source, out in [(io.StringIO('yes\n'), io.StringIO()), (Terminal('\n'), Terminal())]:
            with patch('sys.stdin', source), patch('subprocess.run') as install:
                self.assertFalse(offer_install(error, out))
                self.assertIn('onnx==1.20.0', out.getvalue())
                install.assert_not_called()

    def test_onnx_converter_installs_cpu_vision_without_replacing_torch(self):
        with patch('fxbex.dependencies.installed_version', side_effect={'torch': '2.10.0+cpu', 'torchvision': None}.get):
            vision, converter = installation_commands({'onnx2torch': '1.5.15'})
        self.assertEqual(vision[-3:], ['https://download.pytorch.org/whl/cpu', 'torch==2.10.0+cpu', 'torchvision'])
        self.assertEqual(converter[-3:], ['https://pypi.org/simple', 'onnx2torch==1.5.15', 'torch==2.10.0+cpu'])

    def test_cli_recovers_from_successive_missing_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = SimpleNamespace(
                report_dir=root, root=root, memory={}, manifest={'runtime': {'adapter': 'test'}},
                __enter__=Mock(side_effect=[MissingPackages({'onnx': '1.20.0'}),
                                            ModuleNotFoundError(name='onnx2torch'), None]),
                __exit__=Mock(), predict=Mock(return_value={'output': 1}),
            )
            contract = {'method': 'predict', 'example_input': {}}
            out = io.StringIO()
            with patch('fxbex.settings.runtime_python', return_value=sys.executable), \
                 patch('fxbex.hub.resolve_package', return_value=root/'model.fxbx'), \
                 patch('fxbex.schema.describe', return_value=contract), \
                 patch('fxbex.schema.resolve_sample', return_value={}), \
                 patch('fxbex.runtime.Session', return_value=session), \
                 patch('fxbex.dependencies.installed_version', side_effect={
                     'torch': '2.10.0+cpu', 'torchvision': '0.25.0+cpu',
                 }.get), patch('sys.stdin', Terminal('yes\nyes\n')), redirect_stderr(Terminal()), \
                 redirect_stdout(out), patch('subprocess.run') as install:
                self.assertEqual(main(['run', 'model']), 0)
            self.assertEqual(session.__enter__.call_count, 3)
            self.assertEqual(install.call_count, 2)
            self.assertIn('onnx2torch', install.call_args.args[0])
            self.assertEqual(install.call_args.args[0][:4], [sys.executable, '-m', 'pip', 'install'])
            session.predict.assert_called_once()
