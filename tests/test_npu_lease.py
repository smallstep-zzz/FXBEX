import fcntl
import os
import subprocess,sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.settings import inherited_lease_fd,npu_lock


class Lease(unittest.TestCase):
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

    def test_worker_passes_only_the_validated_lease_and_preserves_parent_lock(self):
        from fxbex.service import PackageService
        real_popen=subprocess.Popen
        code="""import sys
from fxbex.settings import npu_lock
with npu_lock():
 for line in sys.stdin:
  print('@@FXBEX_RESPONSE@@{"ok":true}',flush=True)
"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'npu.lock'
            with path.open('a') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX)
                env={'FXBEX_NPU_LOCK':str(path),'FXBEX_NPU_LOCK_FD':str(owner.fileno())}
                with patch.dict(os.environ,env),patch('fxbex.service.runs_path',return_value=root/'runs'):
                    def child(args,**kwargs):return real_popen([sys.executable,'-u','-c',code],**kwargs)
                    service=PackageService(root/'fixture.fxbex')
                    try:
                        with patch('fxbex.service.subprocess.Popen',side_effect=child) as start:
                            self.assertTrue(service.load()['ok'])
                        self.assertEqual(start.call_args.kwargs['pass_fds'],(owner.fileno(),))
                    finally:service.stop()
                with path.open('a') as other:
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with path.open('a') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_invalid_inherited_fd_is_rejected_before_spawn(self):
        with patch.dict(os.environ,FXBEX_NPU_LOCK_FD='not-a-fd'):
            with self.assertRaisesRegex(RuntimeError,'Invalid inherited'):inherited_lease_fd()
