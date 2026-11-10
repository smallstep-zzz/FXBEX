"""Fake public streams exercise ownership policy, not actual NPU recovery."""
from contextlib import ExitStack
import types
import unittest

from fxbex.native_lifetime import NativeLifetime, finish_close


class NativeLifetimeTests(unittest.TestCase):
    def backend(self):
        sdk = types.ModuleType('fake_sdk')
        stream = types.SimpleNamespace(synchronize=lambda: None)
        sdk.current_stream = lambda device: stream
        torch = types.SimpleNamespace(device=lambda value: value)
        return sdk, torch, stream

    def test_close_synchronizes_captured_stream_not_closing_threads_current(self):
        sdk, torch, submitted = self.backend()
        calls = []
        submitted.synchronize = lambda: calls.append('submitted')
        life = NativeLifetime(sdk, torch, 'furiosa:0', RuntimeError)
        life.begin()
        sdk.current_stream = lambda device: self.fail('must not recapture at close')
        life.synchronize()
        self.assertEqual(calls, ['submitted'])
        self.assertTrue(life.completion_known)

    def test_failed_completion_taints_same_sdk_even_after_later_sync_success(self):
        sdk, torch, stream = self.backend()
        failure = RuntimeError('completion failed')
        stream.synchronize = lambda: (_ for _ in ()).throw(failure)
        life = NativeLifetime(sdk, torch, 'furiosa:0', RuntimeError)
        life.begin()
        with self.assertRaises(RuntimeError) as caught:
            life.synchronize()
        self.assertIs(caught.exception, failure)
        self.assertFalse(life.completion_known)
        stream.synchronize = lambda: None
        life.synchronize()  # Cleanup may retry completion, but cannot clear taint.
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_COMPLETION_UNKNOWN'):
            life.begin()
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_COMPLETION_UNKNOWN'):
            NativeLifetime(sdk, torch, 'furiosa:0', RuntimeError)
        other, _, _ = self.backend()
        NativeLifetime(other, torch, 'furiosa:0', RuntimeError).begin()

    def test_completion_api_is_validated_before_submission(self):
        sdk, torch, _ = self.backend()
        sdk.current_stream = lambda device: object()
        life = NativeLifetime(sdk, torch, 'furiosa:0', ValueError)
        with self.assertRaisesRegex(ValueError, 'completion API'):
            life.begin()
        self.assertTrue(life.completion_known)

    def test_close_keeps_completion_error_before_lease_error(self):
        class FailedRelease:
            def __enter__(self):return self
            def __exit__(self,*args):raise RuntimeError('lease failed')
        stack = ExitStack()
        stack.enter_context(FailedRelease())
        primary = RuntimeError('sync failed')
        with self.assertRaises(RuntimeError) as caught:
            finish_close(stack, primary)
        self.assertIs(caught.exception, primary)
        self.assertIn('lease failed', str(primary.__cause__))


if __name__ == '__main__':
    unittest.main()
