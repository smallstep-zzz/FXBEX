import asyncio
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import importlib

facade = importlib.import_module('fxbex.hug')


class SamplingParams:
    def __init__(self, temperature=1, max_tokens=16, n=1):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n


class Output:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class HugTest(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {
            'furiosa_llm': SimpleNamespace(SamplingParams=SamplingParams),
            'furiosa_llm.outputs': SimpleNamespace(RequestOutput=Output, CompletionOutput=Output),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_native_preserves_constructor_and_instance(self):
        calls = []
        sentinel = object()
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel
        wrapped = facade.hug(original)
        self.assertIs(wrapped('official.fxb', tensor_parallel_size=4), sentinel)
        self.assertEqual(calls, [(('official.fxb',), {'tensor_parallel_size': 4})])
        self.assertIs(facade.hug(wrapped), wrapped)

    def test_offline_native_and_explicit_override(self):
        original = lambda **kwargs: kwargs
        with patch.object(facade, 'urlopen', side_effect=OSError('offline')):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(facade.hug(original)(model='official/model'), {'model': 'official/model'})
        self.assertEqual(facade.hug(original)(model='anything', fxbex=False), {'model': 'anything'})

    def test_exact_catalog_classification(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"models":[{"model":{"name":"qwen","source_repo":"owner/qwen"},"release":{"version":"1"}}]}'
        with patch.object(facade, 'urlopen', return_value=response):
            self.assertTrue(facade._is_package('owner/qwen@1'))
            self.assertTrue(facade._is_package('./owner/qwen@1'))
            self.assertFalse(facade._is_package('owner/qwen@2'))
            self.assertFalse(facade._is_package('unrelated/qwen'))

    def test_constructor_rejects_ignored_options(self):
        wrapped = facade.hug(lambda: None)
        with self.assertRaises(NotImplementedError):
            wrapped('test.fxbex', tensor_parallel_size=2)
        with patch.object(facade.hub, 'resolve_package', return_value='/tmp/model.fxbex'), patch.object(facade, 'inspect', return_value={}):
            with self.assertRaises(NotImplementedError):
                wrapped('test.fxbex', max_model_len=256)

    def test_model_keyword_selects_package(self):
        with patch.object(facade.hub, 'resolve_package', return_value='/tmp/model.fxbex'), patch.object(facade, 'PackageService') as service, patch.object(facade, 'inspect', return_value={}):
            model = facade.hug(lambda **kwargs: None)(model='test.fxbex')
            self.assertIs(model._service, service.return_value)

    def test_dtype_must_match_verified_package_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.fxbex'
            raw=json.dumps({'compute_dtype':'bfloat16'}).encode()
            with zipfile.ZipFile(path,'w') as z:z.writestr('weights_index.json',raw)
            manifest={'files':{'weights_index.json':{'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}
            facade._check_dtype(path,manifest,'bf16')
            with self.assertRaisesRegex(ValueError,'fixed at'):
                facade._check_dtype(path,manifest,'fp16')
            manifest['files']['weights_index.json']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'does not match'):
                facade._check_dtype(path,manifest,'bf16')

    def model(self, context=None, **options):
        service = Mock()
        result = dict(text='answer', token_ids=[31, 32], prompt_token_ids=[4, 5], finish_reason='length')
        service.generate.return_value = result
        with patch.object(facade.hub, 'resolve_package', return_value='/tmp/model.fxbex'), patch.object(facade, 'PackageService', return_value=service), patch.object(facade, 'inspect', return_value={} if context is None else {'context': {'max_tokens': context}}):
            model = facade.hug(lambda: None)('test.fxbex', **options)
        return model, service, result

    def test_package_context_and_token_limits(self):
        for capacity in (512, 262144):
            model, service, _ = self.model(capacity, max_model_len=None)
            self.assertEqual(model.max_model_len, capacity)
            model.generate('hello', SamplingParams(temperature=0, max_tokens=capacity))
            service.generate.assert_called_with(prompt='hello', max_tokens=capacity)
            for tokens in (True, False, 0, -1, capacity + 1, 1.5):
                with self.assertRaises(ValueError):
                    model.generate('hello', SamplingParams(temperature=0, max_tokens=tokens))
        model, _, _ = self.model()
        self.assertEqual(model.max_model_len, 512)
        model, _, _ = self.model(262144, max_model_len=262144)
        self.assertEqual(model.max_model_len, 262144)

    def test_context_override_validation_before_loading(self):
        for max_len in (True, False, 0, -1, 262145, '512'):
            with self.assertRaises(ValueError):
                self.model(262144, max_model_len=max_len)
        with self.assertRaises(NotImplementedError):
            self.model(262144, max_model_len=512)
        for capacity in (True, 0, -1, '262144'):
            with self.assertRaises(ValueError):
                self.model(capacity)

    def test_package_real_output_ids_and_batch(self):
        model, service, result = self.model()
        output = model.generate('hello')
        self.assertEqual(output.prompt_token_ids, [4, 5])
        self.assertEqual(output.outputs[0].token_ids, [31, 32])
        self.assertEqual(len(model.generate(['one', 'two'])), 2)
        service.generate.assert_called_with(prompt='two', max_tokens=16)
        self.assertEqual(len(model.chat([{'role': 'user', 'content': 'hello'}])), 1)
        model.shutdown()
        with self.assertRaises(RuntimeError):
            model.generate('closed')

    def test_unsupported_sampling(self):
        model, service, _ = self.model()
        for params in [SamplingParams(), SamplingParams(temperature=0, n=2)]:
            with self.assertRaises(NotImplementedError):
                model.generate('hello', params)
        service.generate.assert_not_called()
        with self.assertRaises(NotImplementedError):
            model.generate('hello', tokenizer_kwargs={'add_special_tokens':False})

    def test_stream_uses_callback(self):
        model, service, result = self.model()
        def generate(**kwargs):
            kwargs['on_token'](dict(result, text='a', token_ids=[31]))
            return result
        service.generate.side_effect = generate
        async def collect():
            return [item async for item in model.stream_generate('hello')]
        items = asyncio.run(collect())
        self.assertEqual(items, ['a', 'nswer'])

class ClassSemantics(unittest.TestCase):
    def test_class_wrapper_preserves_native_identity_and_methods(self):
        class Native:
            @staticmethod
            def helper():return 'native'
        wrapped=facade.hug(Native)
        obj=wrapped(fxbex=False)
        self.assertIs(type(obj),Native)
        self.assertTrue(isinstance(obj,wrapped))
        self.assertTrue(issubclass(wrapped,Native))
        self.assertEqual(wrapped.helper(),'native')
        self.assertIs(facade.hug(wrapped),wrapped)
        class Derived(wrapped):
            def __init__(self):self.value='derived'
        self.assertEqual(Derived().value,'derived')

class LocalRouting(unittest.TestCase):
    def test_native_fxb_not_intercepted_and_local_version_is(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);native=root/'official.fxb';native.write_bytes(b'native')
            self.assertFalse(facade._is_package(native))
            folder=root/'model'/'opt2';folder.mkdir(parents=True);(folder/'model.fxbex').write_bytes(b'package')
            self.assertTrue(facade._is_package(str(root/'model')+'@opt2'))
