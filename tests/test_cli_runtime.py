import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fxbex.cli import main
from fxbex.settings import runtime_python


class RuntimeCLI(unittest.TestCase):
    def test_run_chat_opens_conversation_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory); calls=[]
            def chat(messages, limit):
                calls.append(list(messages));return {'text':'answer'}
            session=SimpleNamespace(report_dir=report,memory={},root=report,manifest={'runtime':{'adapter':'chat'}},
                __enter__=lambda:None,__exit__=lambda *a:None,chat=chat)
            contract={'method':'chat','example_input':{'messages':[{'role':'user','content':'sample'}]}}
            with patch('fxbex.settings.runtime_python',return_value=sys.executable),patch('fxbex.hub.resolve_package',return_value=report/'model.fxbex'),patch('fxbex.schema.describe',return_value=contract),patch('fxbex.cli.inspect',return_value={}),patch('fxbex.runtime.package_compatibility',return_value=[]),patch('fxbex.runtime.Session',return_value=session),patch('builtins.input',side_effect=['hello','next','/quit']),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run','Model']),0)
            self.assertEqual([x['content'] for x in calls[1]],['hello','answer','next'])

    def test_configured_interpreter_receives_unchanged_arguments_before_resolution(self):
        args=['run','Model','--max-new-tokens','8']
        with patch('fxbex.settings.runtime_python',return_value='/sdk/python'),patch('subprocess.call',return_value=0) as call,patch('fxbex.hub.resolve_package') as resolve:
            self.assertEqual(main(args),0)
        call.assert_called_once_with(['/sdk/python','-m','fxbex',*args])
        resolve.assert_not_called()

    def test_version_does_not_need_sdk_or_catalog(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output),patch('fxbex.hub.resolve_package') as resolve, self.assertRaises(SystemExit) as result:
            main(['--version'])
        self.assertEqual(result.exception.code,0)
        from fxbex import __version__
        self.assertIn('fxbex '+__version__,output.getvalue())
        resolve.assert_not_called()

    def test_runtime_setting_preserves_virtualenv_path_and_env_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);python=root/'venv/python';python.parent.mkdir();python.symlink_to(sys.executable)
            config=root/'config.json';config.write_text(json.dumps({'runtime_python':str(python)}))
            with patch.dict('os.environ',{'FXBEX_CONFIG':str(config),'FXBEX_WORKER_PYTHON':''}):
                self.assertEqual(runtime_python(),str(python))
                with patch.dict('os.environ',{'FXBEX_WORKER_PYTHON':sys.executable}):
                    self.assertEqual(runtime_python(),sys.executable)
                config.write_text(json.dumps({'runtime_python':'relative/python'}))
                with self.assertRaisesRegex(ValueError,'absolute'):runtime_python()

    def test_default_result_is_compact_but_full_diagnostics_are_saved(self):
        for verbose in [False,True]:
            with self.subTest(verbose=verbose),tempfile.TemporaryDirectory() as directory:
                report=Path(directory)
                result={'text':'4','metrics':{'context_limit':262144},'compiler_calls':0,'replay_hits':42,
                        'memory':{'devices':[{'large':'raw-diagnostics'}]}}
                session=SimpleNamespace(report_dir=report,memory={},root=report,manifest={'runtime':{'adapter':'chat'}},
                    __enter__=lambda:None,__exit__=lambda *a:None,chat=lambda *a:dict(result))
                output=io.StringIO()
                args=['run','Model','--sample']+(['--verbose'] if verbose else [])
                contract={'method':'chat','example_input':{'messages':[{'role':'user','content':'2+2?'}]}}
                with contextlib.redirect_stdout(output),patch('fxbex.settings.runtime_python',return_value=sys.executable),patch('fxbex.hub.resolve_package',return_value=report/'model.fxbex'),patch('fxbex.schema.describe',return_value=contract),patch('fxbex.package.inspect',return_value={}),patch('fxbex.cli.inspect',return_value={}),patch('fxbex.runtime.package_compatibility',return_value=[]),patch('fxbex.runtime.Session',return_value=session):
                    self.assertEqual(main(args),0)
                displayed=json.loads(output.getvalue())
                self.assertEqual(displayed['text'],'4')
                self.assertEqual(displayed['metrics']['context_limit'],262144)
                self.assertEqual('memory' in displayed,verbose)
                saved=json.loads(Path(displayed['result_file']).read_text())
                self.assertEqual(saved['memory'],result['memory'])
                self.assertEqual(saved['compiler_calls'],0)
