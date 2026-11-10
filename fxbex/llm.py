"""Chat conveniences for a package-first Model."""
from .model import Model

class LLM(Model):
    def __init__(self, package=None, *, fxbex=None, fxb=None):
        self.history=[]
        super().__init__(package,fxbex=fxbex,fxb=fxb)

    def chat(self, messages, max_new_tokens=32, *, deadline=None):
        """Return a result dict (text, metrics); caller supplies explicit history."""
        return self._call('chat',messages,max_new_tokens,deadline=deadline)

    def generate(self, prompt, max_new_tokens=32):
        """Single-turn generation, returning the same result dict as chat."""
        return self.chat([{'role':'user','content':prompt}],max_new_tokens)

    def converse(self, prompt, max_new_tokens=32):
        """Keep successful turns until reset(); failed calls do not mutate history."""
        with self._guard() as deadline:
            messages=self.history+[{'role':'user','content':prompt}]
            result=self.chat(messages,max_new_tokens,deadline=deadline)
            self.history=messages+[{'role':'assistant','content':result['text']}]
            return result

    def reset(self):
        with self._guard():self.history=[]
