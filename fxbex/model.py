"""Package-first interfaces sharing one process-isolated model lifecycle."""
import threading
import time
from contextlib import contextmanager
from .package import inspect
from .service import PackageService,bounded_lock,request_deadline

class Model:
    def __init__(self, package=None, *, fxbex=None, fxb=None):
        deadline=request_deadline('load')
        paths=[p for p in (package,fxbex,fxb) if p is not None]
        if len(paths)!=1:raise ValueError('Specify one package path: Model(path), fxbex=path, or fxb=path')
        self.manifest=inspect(paths[0])
        self.model=self.manifest['model']['name']
        self._lock=threading.RLock();self._service=PackageService(paths[0])
        self.last_timing={}
        started=time.perf_counter();self._service.load(deadline=deadline)
        self.load_seconds=time.perf_counter()-started

    @contextmanager
    def _guard(self,action='request',deadline=None):
        deadline=request_deadline(action) if deadline is None else deadline
        with bounded_lock(self._lock,deadline):yield deadline

    def _call(self,method,*args,deadline=None):
        started=time.perf_counter();status='failed'
        try:
            with self._guard(deadline=deadline) as deadline:
                result=getattr(self._service,method)(*args,deadline=deadline)
                status='succeeded'
                return result
        finally:
            self.last_timing={'operation':method,'status':status,'wall_seconds':time.perf_counter()-started,'scope':'host request including transfers/waits; not NPU-only'}

    def predict(self,inputs):return self._call('predict',inputs)
    def close(self):
        with self._guard('unload') as deadline:self._service.unload(deadline=deadline)
    def cancel(self):return self._service.cancel()
    unload=close
    def __enter__(self):return self
    def __exit__(self,kind,error,tb):
        try:self.close()
        except BaseException as cleanup:
            if error is not None:raise error.with_traceback(tb) from cleanup
            raise

class YOLO(Model):
    """Detection adapter must accept {'image': server_path, ...options}."""
    def predict(self,image,**options):
        from os import fspath
        return super().predict({'image':fspath(image),**options})

ObjectDetection=YOLO

class ImageGeneration(Model):
    """Prompt-to-image adapter owns sampling, preprocessing and output paths."""
    def generate(self,prompt,**options):
        if not isinstance(prompt,str):raise TypeError('prompt must be text')
        return super().predict({'prompt':prompt,**options})
