"""Public-stream completion and process-local refusal after uncertain completion.

No SDK reset or allocator internals are used. A tainted SDK module stays tainted
until its worker process exits; closing Python references is not device recovery.
"""
import threading
import weakref


_tainted = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def require_healthy_sdk(sdk, error_type):
    with _lock:
        failed = sdk in _tainted
    if failed:
        raise error_type('NATIVE_COMPLETION_UNKNOWN: a previous stream completion failed; '
                         'restart the worker and establish device health before loading again')


def finish_close(stack, error=None):
    """Keep a completion error primary if releasing the lease also fails."""
    try:
        if stack is not None:
            stack.close()
    except BaseException as cleanup:
        if error is not None:
            raise error.with_traceback(error.__traceback__) from cleanup
        raise
    if error is not None:
        raise error.with_traceback(error.__traceback__)


class NativeLifetime:
    def __init__(self, sdk, torch, device, error_type):
        require_healthy_sdk(sdk, error_type)
        self.sdk, self.torch, self.device, self.error_type = sdk, torch, device, error_type
        self.stream = None
        self.completion_known = True

    def begin(self):
        """Capture the submitting thread's stream before its device operations."""
        require_healthy_sdk(self.sdk, self.error_type)
        current = getattr(self.sdk, 'current_stream', None)
        stream = current(self.torch.device(self.device)) if callable(current) else None
        if not callable(getattr(stream, 'synchronize', None)):
            raise self.error_type('Public stream completion API unavailable')
        self.stream = stream

    def synchronize(self):
        """Also callable from close on another thread; never recapture its stream."""
        if self.stream is None:
            return
        try:
            self.stream.synchronize()
        except BaseException:
            self.completion_known = False
            with _lock:
                _tainted[self.sdk] = True
            raise
