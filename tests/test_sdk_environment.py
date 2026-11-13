import io
from pathlib import Path
import unittest
from unittest.mock import patch

from fxbex.dependencies import MissingPackages
from fxbex.sdk_environment import choose_sdk


class SDKEnvironmentTest(unittest.TestCase):
    def test_existing_environment_requires_interactive_selection(self):
        requirements = {'furiosa-torch': '2026.4.0'}
        error = MissingPackages(requirements)
        candidate = Path('/sdk/bin/python')
        argv = ['run', 'FLUX.2-klein-4B', '--sample']
        for interactive in (False, True):
            with self.subTest(interactive=interactive), \
                 patch('fxbex.sdk_environment.sys.platform', 'linux'), \
                 patch('fxbex.sdk_environment.os.uname') as host, \
                 patch('fxbex.sdk_environment.sys.version_info', (3, 14)), \
                 patch('fxbex.sdk_environment.settings', return_value={}), \
                 patch('fxbex.sdk_environment._interpreters', return_value=[candidate]), \
                 patch('fxbex.sdk_environment._probe', return_value=([3, 11], {
                     'fxbx': '0.8.9', **requirements,
                 })), patch('fxbex.sdk_environment.subprocess.call', return_value=0) as launch, \
                 patch('fxbex.sdk_environment.subprocess.run') as install, \
                 patch('fxbex.sdk_environment.sys.stdin') as stdin, \
                 patch('importlib.metadata.version', return_value='0.8.9'):
                host.return_value.machine = 'x86_64'
                stdin.isatty.return_value = interactive
                stdin.readline.return_value = '1\n'
                output = io.StringIO()
                if interactive:
                    self.assertEqual(choose_sdk(error, requirements, argv, output), 0)
                    self.assertEqual(launch.call_args.args[0], [str(candidate), '-m', 'fxbx', *argv])
                else:
                    with self.assertRaises(MissingPackages):
                        choose_sdk(error, requirements, argv, output)
                    launch.assert_not_called()
                install.assert_not_called()
                self.assertIn(str(candidate), output.getvalue())
