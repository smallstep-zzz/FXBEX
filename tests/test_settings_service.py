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

if __name__=='__main__':unittest.main()
