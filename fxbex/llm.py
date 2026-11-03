"""Small process-isolated Python interface for validated chat packages."""
import threading
from .package import inspect
from .service import PackageService


class LLM:
    """Load an FXBEX chat package once. Close explicitly or use a with block.

    fxb is an alias for fxbex; neither converts an official Furiosa FXB.
    Model identity must match the package. No implicit downloads or compilation.
    """
    def __init__(self, model=None, *, fxbex=None, fxb=None):
        if (fxbex is None)==(fxb is None):
            raise ValueError('Specify exactly one of fxbex= or fxb= with an FXBEX package')
        path=fxbex if fxbex is not None else fxb
        # Validate content rather than trusting the filename extension.
        try:manifest=inspect(path)
        except (ValueError, OSError) as exc:
            raise ValueError('Expected an FXBEX package; official .fxb requires furiosa_llm.LLM') from exc
        if model is not None and model!=manifest['model']['name']:
            raise ValueError(f"Model mismatch: requested {model!r}, package contains {manifest['model']['name']!r}")
        self.model=manifest['model']['name'];self.history=[]
        self._lock=threading.RLock();self._service=PackageService(path)
        self._service.load()

    def chat(self, messages, max_new_tokens=32):
        """Return a result dict (text, metrics); caller supplies explicit history."""
        with self._lock:return self._service.chat(messages,max_new_tokens)

    def generate(self, prompt, max_new_tokens=32):
        """Single-turn generation, returning the same result dict as chat."""
        return self.chat([{'role':'user','content':prompt}],max_new_tokens)

    def converse(self, prompt, max_new_tokens=32):
        """Keep successful turns until reset(); failed calls do not mutate history."""
        with self._lock:
            messages=self.history+[{'role':'user','content':prompt}]
            result=self.chat(messages,max_new_tokens)
            self.history=messages+[{'role':'assistant','content':result['text']}]
            return result

    def reset(self):
        with self._lock:self.history=[]

    def close(self):
        with self._lock:self._service.unload()

    unload=close

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
