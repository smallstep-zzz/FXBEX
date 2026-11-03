"""Single-model, serialized localhost service. No automatic compiler or remote code."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time


def serve(session,port):
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,value):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path in ('/status','/v1/models'):
                self.send(200,{'object':'list','data':[{'id':session.manifest['model']['name'],'object':'model'}], 'loaded':session.adapter is not None})
            else:self.send(404,{'error':'Unknown endpoint'})
        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=65536:raise ValueError('Request must be 1..65536 bytes')
                request=json.loads(self.rfile.read(n))
                if request.get('stream'):raise ValueError('Streaming is not supported in v0.1')
                if self.path=='/v1/chat/completions':
                    if request.get('model',session.manifest['model']['name']) != session.manifest['model']['name']:raise ValueError('Requested model is not loaded')
                    result=session.chat(request['messages'],request.get('max_tokens',32))
                    self.send(200,{'id':f'chatcmpl-{time.time_ns()}','object':'chat.completion','created':int(time.time()),'model':session.manifest['model']['name'],
                                  'choices':[{'index':0,'message':{'role':'assistant','content':result['text']},'finish_reason':'stop'}],'fxbex':result})
                elif self.path=='/predict':self.send(200,session.predict(request))
                else:self.send(404,{'error':'Unknown endpoint'})
            except Exception as exc:self.send(400,{'error':str(exc)})
    with HTTPServer(('127.0.0.1',port),Handler) as server:
        print(f'FXB_EX serving http://127.0.0.1:{port} ; Ctrl-C unloads',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
