"""Bounded single-flight worker transport; process exit owns allocation cleanup."""
import atexit
import contextlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time
import weakref
from .settings import inherited_lease_fd, runs_path, runtime_python, worker_timeouts

_services=weakref.WeakSet()
_NO_RESPONSE=object()


class WorkerTimeout(TimeoutError):
    def __init__(self,code,stage):
        self.code,self.stage=code,stage
        super().__init__(f'{code}: worker operation exceeded its deadline ({stage})')


class WorkerCancelled(RuntimeError):
    code='cancelled'
    def __init__(self):super().__init__('cancelled: worker request cancelled; it was not retried')


class WorkerProtocolError(RuntimeError):
    code='protocol_error'


def request_deadline(action='request'):
    return time.monotonic()+worker_timeouts()[action]


def _check(deadline,event,stage,*,queued=False):
    if event is not None and event.is_set():raise WorkerCancelled()
    remaining=deadline-time.monotonic()
    if remaining<=0:raise WorkerTimeout('queue_timeout' if queued else 'worker_timeout',stage)
    return remaining


@contextlib.contextmanager
def bounded_lock(lock,deadline,cancel_event=None):
    """Queue expiry must not stop the worker owned by a different request."""
    if type(deadline) not in (int,float) or not math.isfinite(deadline):
        raise ValueError('deadline must be an absolute finite monotonic time')
    while not lock.acquire(timeout=min(.05,_check(deadline,cancel_event,'queue',queued=True))):pass
    try:
        _check(deadline,cancel_event,'queue',queued=True)
        yield
    finally:lock.release()


class PackageService:
    def __init__(self,package):
        self.package=str(Path(package).resolve())
        self.lock=threading.RLock()
        self._state_lock=threading.Lock();self._stop_lock=threading.Lock()
        self._active=None;self._read_buffer=bytearray()
        self.process=None;self.log=None;self.log_path=None;self.loaded=False
        self.context_limit=512;self.timeouts=worker_timeouts();self.last_failure=None
        self.failed_unreaped=False
        _services.add(self)

    def deadline(self,action='request'):return time.monotonic()+self.timeouts[action]

    def cancel(self):
        """Cancel only the current request without waiting for its I/O lock."""
        with self._state_lock:
            event=self._active['event'] if self._active is not None else None
        if event is not None:event.set()
        return event is not None

    def stop(self):
        self.cancel()
        self._stop_process()

    def _stop_process(self):
        # A second stopper may wait only for the first bounded cleanup, not I/O.
        budget=self.timeouts['terminate']+self.timeouts['kill']
        cleanup_deadline=time.monotonic()+budget
        if not self._stop_lock.acquire(timeout=budget):
            raise RuntimeError('worker_cleanup_timeout: another worker cleanup did not finish')
        try:
            with self._state_lock:
                process,log=self.process,self.log
                self.loaded=False
            first_error=None
            if process is not None:
                try:
                    if process.poll() is None:
                        try:process.terminate()
                        except ProcessLookupError:pass
                        try:process.wait(timeout=max(0,min(self.timeouts['terminate'],cleanup_deadline-time.monotonic())))
                        except subprocess.TimeoutExpired:
                            try:process.kill()
                            except ProcessLookupError:pass
                            process.wait(timeout=max(0,min(self.timeouts['kill'],cleanup_deadline-time.monotonic())))
                    else:process.wait(timeout=max(0,min(self.timeouts['kill'],cleanup_deadline-time.monotonic())))
                except BaseException as error:first_error=error
                if process.poll() is None:
                    self.failed_unreaped=True
                    raise RuntimeError('worker_unreaped: worker termination is unconfirmed; new load is forbidden') from first_error
                for stream in (process.stdin,process.stdout):
                    if stream is not None:
                        try:stream.close()
                        except BaseException as error:
                            if first_error is None:first_error=error
            if log is not None:
                try:log.close()
                except BaseException as error:
                    if first_error is None:first_error=error
            with self._state_lock:
                if self.process is process:
                    self.process=None;self._read_buffer=bytearray();self.failed_unreaped=False
                if self.log is log:self.log=None
            if first_error is not None:raise first_error
        finally:self._stop_lock.release()

    def _start(self):
        with self._state_lock:operation=self._active
        deadline=operation['deadline'] if operation is not None else self.deadline('load')
        event=operation['event'] if operation is not None else None
        # stop cannot report success between child creation and ownership registration.
        with bounded_lock(self._stop_lock,deadline,event):
            if self.failed_unreaped:raise RuntimeError('worker_unreaped: cannot replace a worker whose exit is unconfirmed')
            run=runs_path()/f'ui_{time.time_ns()}';run.mkdir(parents=True)
            self.log_path=run/'worker.log';self.log=self.log_path.open('w')
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8');pass_fds=()
            inherited=inherited_lease_fd()
            if inherited is not None:pass_fds=(inherited,)
            _check(deadline,event,'worker_spawn')
            process=subprocess.Popen([runtime_python(),'-u','-m','fxbex.worker',self.package],env=env,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,pass_fds=pass_fds)
            with self._state_lock:self.process=process;self._read_buffer=bytearray()
            os.set_blocking(process.stdin.fileno(),False);os.set_blocking(process.stdout.fileno(),False)
            _check(deadline,event,'worker_spawn')

    @contextlib.contextmanager
    def _operation(self,action,deadline,cancel_event):
        deadline=self.deadline(action) if deadline is None else deadline
        event=cancel_event if cancel_event is not None else threading.Event()
        with bounded_lock(self.lock,deadline,event):
            operation={'deadline':deadline,'event':event,'action':action}
            with self._state_lock:
                if self._active is not None:
                    raise RuntimeError('worker_reentrant_request: callbacks cannot issue another request on the same worker')
                self._active=operation
            try:yield operation
            finally:
                with self._state_lock:
                    if self._active is operation:self._active=None

    def _line(self,raw,on_progress,on_token):
        line=raw.decode('utf-8')
        if line.startswith('@@FXBEX_PROGRESS@@'):
            event=json.loads(line[len('@@FXBEX_PROGRESS@@'):])
            if on_progress:on_progress(event['phase'],event['label'])
        elif line.startswith('@@FXBEX_TOKEN@@'):
            event=json.loads(line[len('@@FXBEX_TOKEN@@'):])
            if on_token:on_token(event)
        elif line.startswith('@@FXBEX_RESPONSE@@'):
            response=json.loads(line[len('@@FXBEX_RESPONSE@@'):])
            if not isinstance(response,dict):raise WorkerProtocolError('worker_protocol_error: response must be an object')
            if not response.get('ok'):raise RuntimeError(response.get('error','Worker failed'))
            return response
        else:
            self.log.write(line);self.log.flush()
        return _NO_RESPONSE

    def _request(self,request,on_progress=None,on_token=None):
        operation=self._active;deadline,event=operation['deadline'],operation['event']
        process=self.process
        stage='request_write'
        limit=min(64*1024**2,max(4*1024**2,self.context_limit*128))
        payload=(json.dumps(request)+'\n').encode('utf-8')
        if len(payload)>limit:raise ValueError('worker_protocol_error: request exceeds transport frame limit')
        try:
            if process is None:raise RuntimeError('Worker is not running')
            input_fd,output_fd=process.stdin.fileno(),process.stdout.fileno()
            buffer=self._read_buffer;offset=0
            with selectors.DefaultSelector() as selector:
                selector.register(output_fd,selectors.EVENT_READ,'read')
                selector.register(input_fd,selectors.EVENT_WRITE,'write')
                while True:
                    remaining=_check(deadline,event,stage)
                    while b'\n' in buffer:
                        raw,_,rest=buffer.partition(b'\n');buffer[:]=rest
                        if len(raw)>limit:raise WorkerProtocolError('worker_protocol_error: response frame exceeds limit')
                        response=self._line(raw+b'\n',on_progress,on_token)
                        _check(deadline,event,'response_callback')
                        if response is not _NO_RESPONSE:return response
                    if len(buffer)>limit:raise WorkerProtocolError('worker_protocol_error: response frame exceeds limit')
                    for key,mask in selector.select(min(.05,remaining)):
                        _check(deadline,event,stage)
                        if key.data=='write':
                            try:count=os.write(input_fd,payload[offset:])
                            except BlockingIOError:continue
                            offset+=count
                            if offset==len(payload):
                                selector.unregister(input_fd);stage='response_read'
                        else:
                            try:chunk=os.read(output_fd,65536)
                            except BlockingIOError:continue
                            if not chunk:
                                code=process.poll()
                                raise WorkerProtocolError(f'worker_protocol_error: worker stdout ended before response (exit={code}); see {self.log_path}')
                            buffer.extend(chunk)
        except BaseException as error:
            self.last_failure={'stage':stage,'code':getattr(error,'code','worker_failed'),
                'error_type':type(error).__name__,'error':str(error),'completion_known':False,'worker_reaped':False}
            try:self._stop_process()
            except BaseException as cleanup:
                self.last_failure['cleanup_error']=str(cleanup)
                raise error from cleanup
            self.last_failure['worker_reaped']=True
            raise

    def load(self,on_progress=None,*,deadline=None,cancel_event=None):
        with self._operation('load',deadline,cancel_event) as operation:
            with self._state_lock:
                if self.loaded and self.process and self.process.poll() is None:
                    _check(operation['deadline'],operation['event'],'load_commit')
                    return {'ok':True}
            self._stop_process()
            try:
                _check(operation['deadline'],operation['event'],'worker_start')
                self._start()
                with self._state_lock:process=self.process
                result=self._request({'action':'load'},on_progress)
                capacity=result.get('context_limit',512)
                if type(capacity) is not int or capacity<1:raise ValueError('Invalid worker context limit')
                with self._state_lock:
                    _check(operation['deadline'],operation['event'],'load_commit')
                    if process is None or self.process is not process or process.poll() is not None:
                        raise RuntimeError('worker_load_interrupted: loaded worker is no longer alive or owned')
                    self.context_limit=capacity;self.loaded=True
                return result
            except BaseException as error:
                try:self._stop_process()
                except BaseException as cleanup:raise error from cleanup
                raise

    def chat(self,messages,max_new_tokens=32,on_progress=None,*,deadline=None,cancel_event=None):
        with self._operation('request',deadline,cancel_event):
            if not self.loaded:raise RuntimeError('Load a package first')
            if not isinstance(messages,list) or not 1<=len(messages)<=max(16,self.context_limit) or type(max_new_tokens) is not int or not 1<=max_new_tokens<=self.context_limit:
                raise ValueError(f'Bounded text history and 1..{self.context_limit} output tokens required')
            if len(json.dumps(messages).encode())>max(65536,self.context_limit*32):raise ValueError('History exceeds package transport budget')
            return self._request({'action':'chat','messages':messages,'max_new_tokens':max_new_tokens},on_progress)

    def predict(self,inputs,*,deadline=None,cancel_event=None):
        with self._operation('request',deadline,cancel_event):
            if not self.loaded:raise RuntimeError('Load a package first')
            if not isinstance(inputs,dict):raise ValueError('Named inputs must be a dictionary')
            return self._request({'action':'predict','inputs':inputs})

    def generate(self,prompt,max_tokens=16,on_token=None,*,deadline=None,cancel_event=None):
        with self._operation('request',deadline,cancel_event):
            if not self.loaded:raise RuntimeError('Load a package first')
            if type(max_tokens) is not int or not 1<=max_tokens<=self.context_limit:
                raise ValueError(f'max_tokens must be 1..{self.context_limit}')
            if not isinstance(prompt,(str,list)) or len(json.dumps(prompt).encode())>max(65536,self.context_limit*32):
                raise ValueError('Expected bounded prompt text or chat messages')
            return self._request({'action':'generate','prompt':prompt,'max_tokens':max_tokens},on_token=on_token)

    def unload(self,on_progress=None,*,deadline=None,cancel_event=None):
        with self._operation('unload',deadline,cancel_event):
            try:
                if self.process and self.process.poll() is None:result=self._request({'action':'unload'},on_progress)
                else:result={'ok':True}
            except BaseException as error:
                try:self._stop_process()
                except BaseException as cleanup:raise error from cleanup
                raise
            self._stop_process()
            return result


def _stop_all():
    for service in list(_services):
        with contextlib.suppress(BaseException):service.stop()

atexit.register(_stop_all)
