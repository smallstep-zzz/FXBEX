import json
from http.server import HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fxbex.server import serve


class BrowserDemoTest(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def start(self, method, example, *, predict=None, chat=None):
        session = SimpleNamespace(
            adapter=SimpleNamespace(chat=chat),
            manifest={'model': {'name': 'test-model'}, 'files': {'sample.png': {}},
                      'io_contract': {'method': method, 'example_input': example}},
            root=self.root, report_dir=self.root / 'reports', predict=predict, chat=chat,
        )
        session.report_dir.mkdir()
        with patch('fxbex.server.HTTPServer') as factory, patch('builtins.print'):
            serve(session, 0)
        server = HTTPServer(('127.0.0.1', 0), factory.call_args.args[1])
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f'http://127.0.0.1:{server.server_port}'
        return session

    def request(self, path, payload=None, *, demo=True):
        headers = {'Content-Type': 'application/json'}
        if demo:
            headers['X-FXBX-Demo'] = '1'
        data = None if payload is None else json.dumps(payload).encode()
        return urlopen(Request(self.base + path, data, headers), timeout=5)

    def test_browser_chat_and_session_reuse(self):
        sample = {'messages': [{'role': 'user', 'content': 'Hi'}], 'max_new_tokens': 7}
        chat = Mock(side_effect=[{'text': 'one'}, {'text': 'two'}])
        self.start('chat', sample, chat=chat)
        with self.request('/') as page:
            self.assertEqual(page.headers.get_content_type(), 'text/html')
            self.assertIn(b'/demo/run', page.read())
        with self.request('/demo/config') as config:
            self.assertEqual(json.load(config)['example'], sample)
        for expected in ('one', 'two'):
            with self.request('/demo/run', sample) as result:
                self.assertEqual(json.load(result)['result'], {'text': expected})
        self.assertEqual(chat.call_count, 2)
        chat.assert_called_with(sample['messages'], 7)

    def test_decision_values_and_browser_request_guard(self):
        sample = {'state': 'duplicate charge', 'questions': {'team': {'type': 'choice'}}}
        predict = Mock(return_value={'answers': {'team': {'choice': 'billing', 'probabilities': {'billing': 0.9}}}})
        self.start('predict', sample, predict=predict)
        with self.assertRaises(HTTPError) as denied:
            self.request('/demo/run', sample, demo=False)
        self.assertEqual(denied.exception.code, 403)
        with self.request('/demo/run', sample) as result:
            self.assertEqual(json.load(result)['result'], predict.return_value)
        predict.assert_called_once_with(sample)

    def test_image_result_and_packaged_asset(self):
        png = (Path(__file__).resolve().parents[1] / 'examples/web/color-shapes.png').read_bytes()
        sample = {'image': {'$asset': 'sample.png'}}
        (self.root / 'sample.png').write_bytes(png)
        output = self.root / 'reports/result.png'
        predict = Mock(return_value={'image_path': str(output)})
        self.start('predict', sample, predict=predict)
        output.write_bytes(png)
        with self.request('/demo/run', sample) as result:
            image_url = json.load(result)['image_url']
        with self.request(image_url) as result:
            self.assertEqual(result.headers.get_content_type(), 'image/png')
            self.assertEqual(result.read(), png)
        predict.assert_called_once_with({'image': str(self.root / 'sample.png')})
        predict.return_value = {'image_path': str(self.root / 'sample.png')}
        with self.request('/demo/run', sample) as result:
            self.assertNotIn('image_url', json.load(result))
        with self.assertRaises(HTTPError) as invalid:
            self.request('/demo/run', {'image': {'$asset': '../sample.png'}})
        self.assertEqual(invalid.exception.code, 400)
