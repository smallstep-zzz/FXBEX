import fcntl
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxbex.settings import npu_lock


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
