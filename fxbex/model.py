"""Package-first interfaces sharing one process-isolated model lifecycle."""
import threading
import time
from .package import inspect
from .service import PackageService

class Model:
    def __init__(self, package=None, *, fxbex=None, fxb=None):
        paths=[p for p in (package,fxbex,fxb) if p is not None]
        if len(paths)!=1:raise ValueError('Specify one package path: Model(path), fxbex=path, or fxb=path')
        self.manifest=inspect(paths[0])
        self.model=self.manifest['model']['name']
        self._lock=threading.RLock();self._service=PackageService(paths[0])
        self.last_timing={}
        started=time.perf_counter();self._service.load()
        self.load_seconds=time.perf_counter()-started

    def _call(self,method,*args):
        with self._lock:
            started=time.perf_counter()
            status='failed'
            try:
                result=getattr(self._service,method)(*args)
                status='succeeded'
                return result
            finally:
                self.last_timing={'operation':method,'status':status,'wall_seconds':time.perf_counter()-started,'scope':'host request including transfers/waits; not NPU-only'}

    def predict(self,inputs):return self._call('predict',inputs)
    def close(self):
        with self._lock:self._service.unload()
    unload=close
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

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
