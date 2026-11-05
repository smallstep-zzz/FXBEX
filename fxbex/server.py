"""Single-model, serialized localhost service. No automatic compiler or remote code."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time


def serve(session,port,*,invoke=None):
    invoke = invoke or (lambda callback: callback())
    capacity=getattr(getattr(session.adapter,'runtime',None),'capacity',512)
    if type(capacity) is not int or capacity<1:capacity=512
    body_limit=max(65536,capacity*32)
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,value):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path in ('/status','/v1/models'):
                self.send(200,{'object':'list','data':[{'id':session.manifest['model']['name'],'object':'model'}], 'loaded':session.adapter is not None,'context_limit':capacity,'memory':getattr(session,'memory',None)})
            else:self.send(404,{'error':'Unknown endpoint'})
        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=body_limit:raise ValueError(f'Request must be 1..{body_limit} bytes')
                request=json.loads(self.rfile.read(n))
                if self.path=='/v1/chat/completions':
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
        print(f'FXB_EX serving http://127.0.0.1:{port} ; Ctrl-C unloads',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
