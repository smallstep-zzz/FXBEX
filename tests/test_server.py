import io,json,unittest
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
