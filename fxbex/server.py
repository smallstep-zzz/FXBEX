"""Single-model, serialized localhost service. No automatic compiler or remote code."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time

from .web_demo import WebDemo


def serve(session,port,*,invoke=None):
    invoke = invoke or (lambda callback: callback())
    capacity=getattr(getattr(session.adapter,'runtime',None),'capacity',512)
    if type(capacity) is not int or capacity<1:capacity=512
    body_limit=max(65536,capacity*32)
    demo = WebDemo(session)
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,value):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path == '/':
                self.send_demo_bytes(Path(__file__).with_name('web_demo.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/demo/config':
                self.send(200,demo.config)
            elif self.path.startswith('/demo/image/'):
                try:
                    data, mime = demo.image(self.path.removeprefix('/demo/image/'))
                except (KeyError, OSError, ValueError):
                    self.send(404,{'error':'Image unavailable'})
                else:
                    self.send_demo_bytes(data,mime)
            elif self.path in ('/status','/v1/models'):
                self.send(200,{'object':'list','data':[{'id':session.manifest['model']['name'],'object':'model'}], 'loaded':session.adapter is not None,'context_limit':capacity,'memory':getattr(session,'memory',None)})
            else:self.send(404,{'error':'Unknown endpoint'})
        def send_demo_bytes(self,data,mime):
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=body_limit:raise ValueError(f'Request must be 1..{body_limit} bytes')
                request=json.loads(self.rfile.read(n))
                if self.path == '/demo/run':
                    host = self.headers.get('Host', '')
                    if (host.partition(':')[0] not in ('127.0.0.1', 'localhost')
                            or self.headers.get('Origin', 'http://' + host) != 'http://' + host
                            or self.headers.get('X-FXBX-Demo') != '1'):
                        self.send(403,{'error':{'message':'Local browser requests only'}})
                        return
                    self.send(200,demo.run(request,invoke))
                elif self.path=='/v1/chat/completions':
                    supported={'model','messages','max_tokens','max_completion_tokens','stream','temperature','n'}
                    unsupported=sorted(set(request)-supported)
                    if unsupported:raise ValueError('Unsupported request options: '+', '.join(unsupported))
                    if request.get('stream'):raise ValueError('Streaming is not supported by this server')
                    if request.get('temperature',0) != 0:raise ValueError('Only greedy decoding (temperature=0) is supported')
                    if request.get('n',1) != 1:raise ValueError('Only n=1 is supported')
                    if 'max_tokens' in request and 'max_completion_tokens' in request:raise ValueError('Specify only one token limit')
                    if request.get('model',session.manifest['model']['name']) != session.manifest['model']['name']:raise ValueError('Requested model is not loaded')
                    result=invoke(lambda: session.chat(request['messages'],request.get('max_completion_tokens',request.get('max_tokens',32))))
                    result.setdefault('memory',getattr(session,'memory',None))
                    self.send(200,{'id':f'chatcmpl-{time.time_ns()}','object':'chat.completion','created':int(time.time()),'model':session.manifest['model']['name'],
                                  'choices':[{'index':0,'message':{'role':'assistant','content':result['text']},'finish_reason':'stop'}],'fxbex':result})
                elif self.path=='/predict':self.send(200,invoke(lambda: session.predict(request)))
                else:self.send(404,{'error':'Unknown endpoint'})
            except Exception as exc:self.send(400,{'error':{'message':str(exc),'type':'invalid_request_error','param':None,'code':None}})
    with HTTPServer(('127.0.0.1',port),Handler) as server:
        print(f'fxbX 웹 화면: http://127.0.0.1:{server.server_port}/ (Ctrl-C로 종료)',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
