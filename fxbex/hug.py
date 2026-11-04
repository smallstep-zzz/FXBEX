"""Opt-in Furiosa LLM facade for greedy, text-only FXBEX packages.

Native models retain their original constructor and behavior. FXBEX supports
sequential generation, chat, and live asynchronous streaming; advanced native
sampling, tensor parallelism, adapters, and tokenizer access are unsupported.
"""
import asyncio
from functools import wraps
import json
from pathlib import Path
import uuid
import warnings
from typing import TypeVar

_LLMType=TypeVar("_LLMType")
from urllib.request import urlopen

from . import hub
from .service import PackageService


def _is_package(value):
    value = str(value)
    path = Path(value).expanduser()
    if value.endswith('.fxb') or (path.is_file() and not value.endswith('.fxbex')):return False
    identifier=value[2:] if value.startswith('./') else value
    name,separator,version=identifier.partition('@')
    local_path=Path(name).expanduser()/version if separator else path
    if value.endswith('.fxbex') or hub._local_package(local_path) is not None:
        return True
    if value.endswith('.fxb') or path.exists():
        return False
    # Native usage remains available offline; explicit FXBEX resolution still fails loudly.
    try:
        with urlopen(hub.CATALOG, timeout=5) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError('Catalog too large')
        models = json.loads(raw)['models']
        identifier=value[2:] if value.startswith('./') else value
        name, separator, version = identifier.partition('@')
        return any(name in (item['model']['name'], item['model']['source_repo'])
                   and (not separator or version == item['release']['version']) for item in models)
    except Exception as exc:
        warnings.warn(f'FuryAwesome catalog unavailable; using native LLM ({exc}). Use fxbex=True to require FXBEX.', RuntimeWarning, stacklevel=3)
        return False


def hug(original_class: type[_LLMType]) -> type[_LLMType]:
    """Wrap an official LLM constructor without modifying its native path.

    ``fxbex=True`` forces package resolution; ``fxbex=False`` forces native.
    Otherwise package paths and exact FuryAwesome catalog IDs select FXBEX.
    """
    if getattr(original_class, '_fxbex_hug', False):
        return original_class

    @wraps(original_class)
    def wrapped(*args, **kwargs):
        options = dict(kwargs)
        explicit = options.pop('fxbex', None)
        if explicit is not None and type(explicit) is not bool:
            raise TypeError('fxbex must be True or False')
        model = args[0] if args else options.get('model_id_or_path',options.get('model'))
        if explicit is False or (explicit is None and (model is None or not _is_package(model))):
            return original_class(*args, **options)
        if model is None:
            raise TypeError('An FXBEX model path or catalog ID is required')
        if len(args) > 1:
            raise NotImplementedError('FXBEX accepts only model as a positional constructor argument')
        options.pop('model', None)
        options.pop('model_id_or_path', None)
        return _PackageLLM(model, **options)

    wrapped._fxbex_hug = True
    if isinstance(original_class,type):
        class HugMeta(type(original_class)):
            def __call__(cls,*args,**kwargs):
                if cls is wrapped_class:return wrapped(*args,**kwargs)
                return super().__call__(*args,**kwargs)
            def __instancecheck__(cls,instance):
                if cls is not wrapped_class:return super().__instancecheck__(instance)
                return isinstance(instance,(original_class,_PackageLLM))
        wrapped_class=HugMeta(original_class.__name__,(original_class,),{
            '__module__':original_class.__module__,'__doc__':original_class.__doc__,
            '__wrapped__':original_class,'_fxbex_hug':True})
        return wrapped_class
    return wrapped


def _max_tokens(params):
    if params is None:
        return 16
    from furiosa_llm import SamplingParams
    defaults = SamplingParams()
    fields = set(vars(defaults)) | set(vars(params))
    for name in fields - {'temperature', 'max_tokens'}:
        if getattr(params, name, None) != getattr(defaults, name, None):
            raise NotImplementedError(f'FXBEX does not support SamplingParams.{name}; use its default')
    if getattr(params, 'temperature', None) != 0:
        raise NotImplementedError('FXBEX supports greedy decoding only; set temperature=0')
    value = getattr(params, 'max_tokens', 16)
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError('FXBEX max_tokens must be an integer in 1..256')
    return value


def _output(result, request_id, finished=True):
    from furiosa_llm.outputs import CompletionOutput, RequestOutput
    ids = result['token_ids']
    prompt_ids = result['prompt_token_ids']
    completion = CompletionOutput(index=0, text=result['text'], token_ids=ids,
                                  logprobs=None, finish_reason=result.get('finish_reason') if finished else None)
    return RequestOutput(request_id=request_id, prompt=result.get('prompt'),
                         prompt_token_ids=prompt_ids, prompt_logprobs=None,
                         outputs=[completion], finished=finished,
                         num_output_tokens=len(ids), num_cached_tokens=result.get('num_cached_tokens', 0),
                         num_prompt_tokens=len(prompt_ids))


class _PackageLLM:
    def __init__(self, model, **options):
        max_len = options.pop('max_model_len', 512)
        if max_len is None:max_len=512
        native_defaults={'fxb':None,'revision':None,'devices':None,'speculative_config':None,'speculative_method':None,'draft_model_id_or_path':None,'draft_fxb_path':None,'num_speculative_tokens':None,'data_parallel_size':None,'pipeline_parallel_size':None,'max_io_memory_mb':None,'scheduler_config':None,'logger_config':None,'tokenizer':None,'tokenizer_mode':'auto','served_model_name':None,'kv_transfer_config':None,'mm_processor_kwargs':None,'structured_outputs_backend':'auto'}
        options={k:v for k,v in options.items() if k not in native_defaults or v!=native_defaults[k]}
        if type(max_len) is not int or max_len != 512:
            raise NotImplementedError('FXBEX currently requires max_model_len=512; smaller context overrides are unsupported')
        if options:
            raise NotImplementedError('Unsupported FXBEX constructor options: ' + ', '.join(sorted(options)))
        self._service = PackageService(hub.resolve_package(model))
        self.max_model_len=512
        self.is_generative_model=True
        self._closed = False
        self.load_result=self._service.load()

    def _check(self):
        if self._closed:
            raise RuntimeError('FXBEX model has been shut down')

    def generate(self, prompts, sampling_params=None, prompt_token_ids=None, tokenizer_kwargs=None):
        self._check()
        if prompt_token_ids is not None or tokenizer_kwargs not in (None,{}):
            raise NotImplementedError('FXBEX accepts text prompts without prompt_token_ids or tokenizer_kwargs')
        limit = _max_tokens(sampling_params)
        single = isinstance(prompts, str)
        batch = [prompts] if single else prompts
        if not isinstance(batch, list) or not all(isinstance(item, str) for item in batch):
            raise TypeError('prompts must be a string or list of strings')
        results = [_output(self._service.generate(prompt=item, max_tokens=limit), str(uuid.uuid4())) for item in batch]
        return results[0] if single else results

    def chat(self, messages, sampling_params=None, **kwargs):
        self._check()
        defaults={'chat_template':None,'chat_template_content_format':'string','add_generation_prompt':True,'continue_final_message':False,'tools':None,'chat_template_kwargs':None}
        kwargs={k:v for k,v in kwargs.items() if k not in defaults or v!=defaults[k]}
        if kwargs:
            raise NotImplementedError('Unsupported FXBEX chat options: ' + ', '.join(sorted(kwargs)))
        limit = _max_tokens(sampling_params)
        if not isinstance(messages, list):
            raise TypeError('messages must be a list')
        batch = messages if messages and isinstance(messages[0], list) else [messages]
        return [_output(self._service.generate(prompt=item, max_tokens=limit), str(uuid.uuid4())) for item in batch]

    async def stream_generate(self, prompt, sampling_params=None, **kwargs):
        self._check()
        if kwargs:
            raise NotImplementedError('Unsupported FXBEX streaming options: ' + ', '.join(sorted(kwargs)))
        if not isinstance(prompt, str):
            raise TypeError('FXBEX streaming accepts one text prompt')
        limit = _max_tokens(sampling_params)
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        emitted = ''
        def on_token(event):
            loop.call_soon_threadsafe(queue.put_nowait, event)
        def run():
            try:
                result = self._service.generate(prompt=prompt, max_tokens=limit, on_token=on_token)
                loop.call_soon_threadsafe(queue.put_nowait, (True, result))
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, (False, exc))
        worker = asyncio.create_task(asyncio.to_thread(run))
        try:
            while True:
                item = await queue.get()
                if isinstance(item, tuple):
                    if not item[0]:
                        raise item[1]
                    remaining = item[1]['text'][len(emitted):]
                    if remaining:
                        yield remaining
                    break
                text = item['text']
                if not text.startswith(emitted):raise RuntimeError('Tokenizer rewrote emitted text; incremental append stream unavailable')
                delta = text[len(emitted):]
                emitted = text
                if delta:
                    yield delta
        finally:
            # Cancellation must not release the service while its request runs.
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await worker
                raise

    def shutdown(self):
        self._service.stop()
        self._closed = True

    close = shutdown
    unload = shutdown

    def __enter__(self):return self
    def __exit__(self,*args):self.shutdown()

    def encode(self,*args,**kwargs):raise NotImplementedError('FXBEX text-generation package does not provide embeddings/encode')
    embed=encode
    def score(self,*args,**kwargs):raise NotImplementedError('FXBEX text-generation package does not provide scoring')

    def get_default_sampling_params(self):
        from furiosa_llm import SamplingParams
        return SamplingParams(temperature=0, max_tokens=16)

    def get_tokenizer(self):
        raise NotImplementedError('FXBEX tokenizer lives in its worker; use generate or chat')
