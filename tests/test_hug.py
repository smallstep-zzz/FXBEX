import asyncio
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
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
            service.generate.assert_called_with(prompt='hello', max_tokens=capacity,deadline=service.deadline.return_value)
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
        service.generate.assert_called_with(prompt='two', max_tokens=16,deadline=service.deadline.return_value)
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

    def test_batch_calls_share_one_absolute_deadline(self):
        model, service, _ = self.model()
        service.deadline.return_value=123.0
        model.generate(['first','second'])
        self.assertEqual(service.deadline.call_count,1)
        self.assertEqual([call.kwargs['deadline'] for call in service.generate.call_args_list],[123.0,123.0])

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

    def test_stream_cancellation_signals_only_its_request(self):
        model, service, _ = self.model()
        entered, finished = threading.Event(), threading.Event()
        events = []
        service.deadline.return_value = 123.0
        def generate(**kwargs):
            events.append(kwargs['cancel_event'])
            self.assertEqual(kwargs['deadline'],123.0)
            entered.set()
            if not kwargs['cancel_event'].wait(2):
                raise AssertionError('stream did not cancel its request')
            finished.set()
            raise RuntimeError('request cancelled')
        service.generate.side_effect = generate
        async def collect():
            stream = model.stream_generate('hello')
            call = asyncio.create_task(anext(stream))
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            call.cancel()
            with self.assertRaises(asyncio.CancelledError):await call
            await stream.aclose()
        asyncio.run(collect())
        self.assertTrue(finished.is_set())
        self.assertEqual(len(events),1)
        self.assertTrue(events[0].is_set())
        service.stop.assert_not_called()

    def test_context_cleanup_preserves_primary_error_and_closes_facade(self):
        model, service, _ = self.model()
        primary, cleanup = RuntimeError('primary inference failure'), RuntimeError('stop failure')
        service.stop.side_effect = cleanup
        with self.assertRaisesRegex(RuntimeError,'primary inference failure') as failure:
            with model:raise primary
        self.assertIs(failure.exception,primary)
        self.assertIs(failure.exception.__cause__,cleanup)
        with self.assertRaisesRegex(RuntimeError,'shut down'):model.generate('again')

    def test_context_cleanup_error_is_visible_without_primary_error(self):
        model, service, _ = self.model()
        service.stop.side_effect = RuntimeError('stop failure')
        with self.assertRaisesRegex(RuntimeError,'stop failure'):
            with model:pass

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
    def archive(self,path,producer=None,manifest=None):
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('manifest.json',json.dumps({'format_version':2} if manifest is None else manifest))
            if producer is not None:
                archive.writestr(facade.PRODUCER_MEMBER,json.dumps(producer))

    def test_native_fxb_not_intercepted_and_local_version_is(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);native=root/'official.fxb';self.archive(native)
            self.assertFalse(facade._is_package(native))
            folder=root/'model'/'opt2';folder.mkdir(parents=True);(folder/'model.fxbex').write_bytes(b'package')
            self.assertTrue(facade._is_package(str(root/'model')+'@opt2'))

    def test_local_package_manifest_wins_over_fxb_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'renamed.fxb'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('manifest.json',json.dumps({'format':'FXB_EX','schema_version':1}))
            with patch.object(facade,'urlopen',side_effect=AssertionError('local routing must not query catalog')):
                self.assertTrue(facade._is_package(path))
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('manifest.json',json.dumps({'format_version':2}))
            self.assertFalse(facade._is_package(path))

    def test_explicit_package_suffix_keeps_package_loader_guidance(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'native.fxbex';self.archive(path)
            self.assertTrue(facade._is_package(path))
            original=Mock();original._fxbex_hug=False
            with patch.object(facade,'_PackageLLM') as package:
                facade.hug(original)(path)
                package.assert_called_once_with(path)
            original.assert_not_called()

    def test_tcl_plan_is_rejected_before_official_constructor(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'custom.fxb'
            original=Mock();original._fxbex_hug=False
            wrapped=facade.hug(original)
            for producer in ({'execution_plan':{}},{'graph_plan':{}},{'explicit_consumer_required':True}):
                self.archive(path,producer)
                calls=[lambda:wrapped(path),lambda:wrapped(model=path),
                       lambda:wrapped(model_id_or_path=path),lambda:wrapped(d,fxb=path)]
                with patch.object(facade,'urlopen',side_effect=AssertionError('no local catalog request')):
                    for call in calls:
                        with self.assertRaisesRegex(ValueError,'TCL_NATIVE_FXB_REQUIRES_GRAPH_LOADER') as failure:call()
                        self.assertIn('weights=',str(failure.exception))
            original.assert_not_called()

    def test_official_fxb_option_preserves_constructor_without_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'official.fxb';self.archive(path,{'operation_source':'official_model_build'})
            original=Mock();original._fxbex_hug=False
            with patch.object(facade,'urlopen',side_effect=AssertionError('no local catalog request')):
                facade.hug(original)('owner/model',fxb=path,tensor_parallel_size=8)
            original.assert_called_once_with('owner/model',fxb=path,tensor_parallel_size=8)

    def test_explicit_native_override_keeps_existing_passthrough(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'custom.fxb';self.archive(path,{'execution_plan':{}})
            original=Mock();original._fxbex_hug=False
            facade.hug(original)(d,fxb=path,fxbex=False)
            original.assert_called_once_with(d,fxb=path)

    def test_package_cannot_be_passed_as_official_fxb_option(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'renamed.fxb';self.archive(path,manifest={'format':'FXB_EX','schema_version':1})
            original=Mock();original._fxbex_hug=False
            with self.assertRaisesRegex(ValueError,'FXB_EX_REQUIRES_PACKAGE_LOADER'):
                facade.hug(original)(d,fxb=path)
            original.assert_not_called()

    def test_invalid_local_metadata_never_falls_through_to_official_loader(self):
        import warnings
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'invalid.fxb'
            original=Mock();original._fxbex_hug=False
            wrapped=facade.hug(original)
            for invalid in ('not zip','not json','duplicate manifest','duplicate producer','duplicate key','nonfinite','bad producer','oversized'):
                with self.subTest(invalid=invalid),warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if invalid=='not zip':path.write_bytes(b'not an FXB')
                    else:
                        with zipfile.ZipFile(path,'w') as archive:
                            manifest='not json' if invalid=='not json' else '{"format_version":2}'
                            if invalid=='duplicate key':manifest='{"format_version":2,"format_version":2}'
                            if invalid=='nonfinite':manifest='{"format_version":2,"value":NaN}'
                            if invalid=='oversized':manifest=' '*(facade.MAX_JSON_SIZE+1)
                            archive.writestr('manifest.json',manifest)
                            if invalid=='duplicate manifest':archive.writestr('manifest.json',manifest)
                            if invalid=='bad producer':archive.writestr(facade.PRODUCER_MEMBER,'[]')
                            if invalid=='duplicate producer':
                                archive.writestr(facade.PRODUCER_MEMBER,'{}')
                                archive.writestr(facade.PRODUCER_MEMBER,'{}')
                    with patch.object(facade,'urlopen',side_effect=AssertionError('no local catalog request')):
                        with self.assertRaisesRegex(ValueError,'INVALID_LOCAL_FXB_METADATA'):wrapped(path)
            original.assert_not_called()
