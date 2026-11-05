import io,json,unittest,threading
from http.server import HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock,patch
from fxbex.server import serve

class ServerContext(unittest.TestCase):
    def request(self,capacity):
        session=SimpleNamespace(adapter=SimpleNamespace(runtime=SimpleNamespace(capacity=capacity)),
            manifest={'model':{'name':'test'}},memory=None,chat=Mock(return_value={'text':'ok'}))
        with patch('fxbex.server.HTTPServer') as server,patch('builtins.print'):
            serve(session,8000)
        handler=server.call_args.args[1]
        body=json.dumps({'messages':[{'role':'user','content':'a'*70000}]}).encode()
        responses=[]
        request=SimpleNamespace(headers={'Content-Length':str(len(body))},rfile=io.BytesIO(body),
            path='/v1/chat/completions',send=lambda code,result:responses.append((code,result)))
        handler.do_POST(request)
        return session,responses

    def test_long_package_accepts_body_above_legacy_limit(self):
        session,response=self.request(262144)
        self.assertEqual(response[0][0],200)
        session.chat.assert_called_once()

    def test_legacy_package_retains_bounded_body(self):
        session,response=self.request(512)
        self.assertEqual(response[0][0],400)
        session.chat.assert_not_called()

    def test_requests_use_diagnostic_wrapper(self):
        session=SimpleNamespace(adapter=SimpleNamespace(runtime=SimpleNamespace(capacity=512)),
            manifest={'model':{'name':'test'}},memory=None,chat=Mock(return_value={'text':'synthetic'}))
        invoke=Mock(side_effect=lambda callback:callback())
        with patch('fxbex.server.HTTPServer') as server,patch('builtins.print'):
            serve(session,8000,invoke=invoke)
        handler=server.call_args.args[1]
        body=json.dumps({'model':'test','messages':[{'role':'user','content':'x'}]}).encode()
        responses=[]
        request=SimpleNamespace(headers={'Content-Length':str(len(body))},rfile=io.BytesIO(body),
            path='/v1/chat/completions',send=lambda code,result:responses.append(code))
        handler.do_POST(request)
        self.assertEqual(responses,[200])
        invoke.assert_called_once()

    def test_openai_client_roundtrip_and_rejects_unsupported_options(self):
        try:
            from openai import OpenAI, BadRequestError
        except ImportError:
            self.skipTest('openai optional dependency is unavailable')
        session=SimpleNamespace(adapter=SimpleNamespace(runtime=SimpleNamespace(capacity=512)),
            manifest={'model':{'name':'test'}},memory=None,chat=Mock(return_value={'text':'synthetic'}))
        with patch('fxbex.server.HTTPServer') as factory, patch('builtins.print'):
            serve(session,8000)
        handler=factory.call_args.args[1]
        server=HTTPServer(('127.0.0.1',0),handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(thread.join,5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client=OpenAI(api_key='local',base_url=f'http://127.0.0.1:{server.server_port}/v1')
        self.addCleanup(client.close)
        messages=[{'role':'user','content':'test'}]
        response=client.chat.completions.create(model='test',messages=messages,max_completion_tokens=7)
        self.assertEqual(response.choices[0].message.content,'synthetic')
        session.chat.assert_called_once_with(messages,7)
        with self.assertRaises(BadRequestError):
            client.chat.completions.create(model='test',messages=messages,temperature=0.8)
        session.chat.assert_called_once()
