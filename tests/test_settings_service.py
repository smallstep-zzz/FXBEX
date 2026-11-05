import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.settings import lock_path
from fxbex.service import PackageService
from fxbex.runtime import resolve_adapter


class IntegrationContracts(unittest.TestCase):
    def test_shared_lock_config_and_override(self):
        with tempfile.TemporaryDirectory() as t:
            config=Path(t)/'config.json'
            config.write_text(json.dumps({'npu_lock':str(Path(t)/'shared.lock')}))
            with patch.dict('os.environ',{'FXBEX_CONFIG':str(config)},clear=True):
                self.assertEqual(lock_path(),(Path(t)/'shared.lock').resolve())
                with patch.dict('os.environ',{'FXBEX_NPU_LOCK':str(Path(t)/'override.lock')}):
                    self.assertEqual(lock_path(),(Path(t)/'override.lock').resolve())

    def test_failed_worker_start_closes_log(self):
        service=PackageService('/tmp/example.fxbex')
        log=io.StringIO()
        def fail():
            service.log=log
            raise OSError('spawn failed')
        with patch.object(service,'_start',side_effect=fail):
            with self.assertRaisesRegex(OSError,'spawn failed'):service.load()
        self.assertTrue(log.closed)
        self.assertFalse(service.loaded)

    def test_missing_or_duplicate_plugins_are_rejected(self):
        with patch('importlib.metadata.entry_points',return_value=[]):
            with self.assertRaisesRegex(ValueError,'exactly one'):resolve_adapter('missing')
        class Plugin:
            name='duplicate'
        with patch('importlib.metadata.entry_points',return_value=[Plugin(),Plugin()]):
            with self.assertRaisesRegex(ValueError,'found 2'):resolve_adapter('duplicate')

class ContextProtocol(unittest.TestCase):
    def test_loaded_capacity_controls_output_and_transport(self):
        service=PackageService('/tmp/fixture.fxbex')
        with patch.object(service,'_start'), patch.object(service,'_request',return_value={'ok':True,'context_limit':262144}):
            service.load()
            self.assertEqual(service.context_limit,262144)
            service.generate('a'*70000,max_tokens=1024)
            service.chat([{'role':'user','content':'a'*70000}],max_new_tokens=1024)
            with self.assertRaises(ValueError):service.generate('a',max_tokens=262145)
            with self.assertRaises(ValueError):service.generate('a',max_tokens=True)
        service.stop()


class TokenProtocol(unittest.TestCase):
    def test_tokens_arrive_before_final_response(self):
        from types import SimpleNamespace
        service=PackageService('/tmp/fixture.fxbex');service.loaded=True
        service.process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO('@@FXBEX_TOKEN@@{"text":"a","token_ids":[1]}\n@@FXBEX_RESPONSE@@{"ok":true,"text":"ab"}\n'))
        service.log=io.StringIO();events=[]
        result=service.generate('prompt',max_tokens=2,on_token=events.append)
        self.assertEqual(events,[{'text':'a','token_ids':[1]}]);self.assertEqual(result['text'],'ab')
        service.process=None;service.log=None

if __name__=='__main__':unittest.main()
