import fcntl
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.settings import npu_lock


class Lease(unittest.TestCase):
    def test_bundled_session_shares_lease_and_releases_it_on_exit(self):
        from fxbex.runtime import Session

        def adapter(_root, _report, stack, *, weights_dir):
            stack.enter_context(npu_lock())
            if fail:
                raise ValueError('adapter failed')
            return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps({'runtime': {'adapter': 'test_bundle'}}))
            with patch.dict(os.environ, {'FXBEX_NPU_LOCK': str(root/'device.lock')}), \
                 patch('fxbex.runtime.extract', return_value=nullcontext(root)), \
                 patch('fxbex.runtime.check_compatibility'), patch('fxbex.validation.preflight'), \
                 patch('fxbex.memory.analyze', return_value={}), patch('fxbex.execution.MemoryTelemetry'), \
                 patch('fxbex.bundled.activate', return_value=nullcontext(adapter)):
                os.environ.pop('FXBEX_NPU_LOCK_FD', None)
                for fail in (False, True):
                    session = Session('unused', report_dir=root/'reports')
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'adapter failed'):
                            with session:
                                self.fail('construction should have failed')
                    else:
                        with session:
                            # A new acquisition outside adapter construction still contends.
                            with self.assertRaises(BlockingIOError):
                                with npu_lock():
                                    self.fail('a second owner should not acquire the device')
                    with npu_lock():
                        pass

    def test_inherited_descriptor_is_validated_and_parent_remains_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'npu.lock'
            with path.open('a') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX)
                with patch.dict(os.environ,FXBEX_NPU_LOCK=str(path),FXBEX_NPU_LOCK_FD=str(owner.fileno())):
                    with npu_lock():pass
                with path.open('a') as other:
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                wrong=Path(directory)/'wrong';wrong.touch()
                with patch.dict(os.environ,FXBEX_NPU_LOCK=str(wrong),FXBEX_NPU_LOCK_FD=str(owner.fileno())):
                    with self.assertRaises(RuntimeError):
                        with npu_lock():pass
