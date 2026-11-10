import io
import json
import os,subprocess,sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.settings import lock_path,worker_timeouts
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
        from types import SimpleNamespace
        service=PackageService('/tmp/fixture.fxbex')
        process=SimpleNamespace(poll=lambda:None)
        def start():service.process=process
        with patch.object(service,'_start',side_effect=start),patch.object(service,'_stop_process'),patch.object(service,'_request',return_value={'ok':True,'context_limit':262144}):
            service.load()
            self.assertEqual(service.context_limit,262144)
            service.generate('a'*70000,max_tokens=1024)
            service.chat([{'role':'user','content':'a'*70000}],max_new_tokens=1024)
            with self.assertRaises(ValueError):service.generate('a',max_tokens=262145)
            with self.assertRaises(ValueError):service.generate('a',max_tokens=True)
        service.process=None
        service.stop()


class TokenProtocol(unittest.TestCase):
    def test_tokens_arrive_before_final_response(self):
        service=PackageService('/tmp/fixture.fxbex');service.loaded=True
        code="import sys;sys.stdin.readline();print('@@FXBEX_TOKEN@@{\"text\":\"a\",\"token_ids\":[1]}');print('@@FXBEX_RESPONSE@@{\"ok\":true,\"text\":\"ab\"}',flush=True)"
        service.process=subprocess.Popen([sys.executable,'-u','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        self.addCleanup(service.stop)
        os.set_blocking(service.process.stdin.fileno(),False);os.set_blocking(service.process.stdout.fileno(),False)
        service.log=io.StringIO();events=[]
        result=service.generate('prompt',max_tokens=2,on_token=events.append)
        self.assertEqual(events,[{'text':'a','token_ids':[1]}]);self.assertEqual(result['text'],'ab')


class TimeoutPolicy(unittest.TestCase):
    def test_timeouts_are_host_owned_finite_positive_values(self):
        with patch('fxbex.settings.settings',return_value={}):
            self.assertEqual(worker_timeouts(),{'load':600.,'request':300.,'unload':30.,'terminate':15.,'kill':5.})
        with patch('fxbex.settings.settings',return_value={'worker_timeouts':{'request':12}}):
            self.assertEqual(worker_timeouts()['request'],12.)
        for value in (None,[],{'unknown':1},{'request':0},{'request':True},{'request':float('nan')},{'kill':float('inf')},{'load':1e100}):
            with self.subTest(value=value),patch('fxbex.settings.settings',return_value={'worker_timeouts':value}):
                with self.assertRaises(ValueError):worker_timeouts()

if __name__=='__main__':unittest.main()
