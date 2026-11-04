"""Standalone subprocess client; closing the worker releases its NPU allocation."""
import atexit
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref
from .settings import runs_path

_services=weakref.WeakSet()


class PackageService:
    def __init__(self,package):
        self.package=str(Path(package).resolve())
        self.lock=threading.RLock()
        self.process=None;self.log=None;self.log_path=None;self.loaded=False
        _services.add(self)

    def stop(self):
        with self.lock:
            if self.process:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                for stream in (self.process.stdin,self.process.stdout):
                    if stream:
                        with contextlib.suppress(OSError):stream.close()
                self.process=None
            if self.log:self.log.close();self.log=None
            self.loaded=False

    def _start(self):
        run=runs_path()/f'ui_{time.time_ns()}';run.mkdir(parents=True)
        self.log_path=run/'worker.log';self.log=self.log_path.open('w')
        env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        worker_python=os.environ.get('FXBEX_WORKER_PYTHON',sys.executable)
        if not Path(worker_python).is_absolute():raise ValueError('FXBEX_WORKER_PYTHON must be an absolute interpreter path')
        self.process=subprocess.Popen([worker_python,'-u','-m','fxbex.worker',self.package],env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)

    def _request(self,request,on_progress=None,on_token=None):
        try:
            self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
            for line in self.process.stdout:
                if line.startswith('@@FXBEX_PROGRESS@@'):
                    event=json.loads(line[len('@@FXBEX_PROGRESS@@'):])
                    if on_progress:on_progress(event['phase'],event['label'])
                elif line.startswith('@@FXBEX_TOKEN@@'):
                    event=json.loads(line[len('@@FXBEX_TOKEN@@'):])
                    if on_token:on_token(event)
                elif line.startswith('@@FXBEX_RESPONSE@@'):
                    response=json.loads(line[len('@@FXBEX_RESPONSE@@'):])
                    if not response.get('ok'):raise RuntimeError(response.get('error','Worker failed'))
                    return response
                else:self.log.write(line);self.log.flush()
            code=self.process.wait()
            raise RuntimeError(f'FXBEX worker exited ({code}); see {self.log_path}')
        except BaseException:self.stop();raise

    def load(self,on_progress=None):
        with self.lock:
            if self.loaded and self.process and self.process.poll() is None:return {'ok':True}
            self.stop()
            try:
                self._start()
                result=self._request({'action':'load'},on_progress);self.loaded=True
                return result
            except BaseException:self.stop();raise

    def chat(self,messages,max_new_tokens=32,on_progress=None):
        with self.lock:
            if not self.loaded:raise RuntimeError('Load a package first')
            if not isinstance(messages,list) or len(messages)>16 or type(max_new_tokens) is not int or not 1<=max_new_tokens<=256:
                raise ValueError('Bounded text history and 1..256 output tokens required')
            if len(json.dumps(messages).encode())>65536:raise ValueError('History exceeds 64KiB')
            return self._request({'action':'chat','messages':messages,'max_new_tokens':max_new_tokens},on_progress)

    def predict(self,inputs):
        with self.lock:
            if not self.loaded:raise RuntimeError('Load a package first')
            if not isinstance(inputs,dict):raise ValueError('Named inputs must be a dictionary')
            return self._request({'action':'predict','inputs':inputs})

    def generate(self, prompt, max_tokens=16, on_token=None):
        with self.lock:
            if not self.loaded:raise RuntimeError('Load a package first')
            if type(max_tokens) is not int or not 1<=max_tokens<=256:
                raise ValueError('max_tokens must be 1..256')
            if not isinstance(prompt,(str,list)) or len(json.dumps(prompt).encode())>65536:
                raise ValueError('Expected bounded prompt text or chat messages')
            return self._request({'action':'generate','prompt':prompt,'max_tokens':max_tokens},on_token=on_token)

    def unload(self,on_progress=None):
        with self.lock:
            try:
                if self.process and self.process.poll() is None:return self._request({'action':'unload'},on_progress)
                return {'ok':True}
            finally:self.stop()


def _stop_all():
    for service in list(_services):service.stop()

atexit.register(_stop_all)
