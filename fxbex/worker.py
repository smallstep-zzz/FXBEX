"""Persistent stdio package worker using the existing bounded service protocol."""
import contextlib
import ctypes
import json
import os
import signal
import sys
import traceback
from .runtime import Session


def main():
    parent=os.getppid()
    if sys.platform.startswith('linux'):
        if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('parent-death signal failed')
        if os.getppid()!=parent or parent==1:raise RuntimeError('Parent exited')
    protocol=sys.stdout;session=None
    try:
        for line in sys.stdin:
            request={}
            try:
                request=json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    if request['action']=='load':
                        if session:raise RuntimeError('Package already loaded')
                        print('@@FXBEX_PROGRESS@@'+json.dumps({'phase':'loading','label':'FXB_EX: 무결성 검사·가중치 적재·커널 복원 중'}),file=protocol,flush=True)
                        candidate=Session(sys.argv[1]);candidate.__enter__();session=candidate
                        result={'ok':True,'metrics':getattr(session.adapter,'acceptance',{}),'report_dir':str(session.report_dir),'context_limit':getattr(getattr(session.adapter,'runtime',None),'capacity',512),'memory':session.memory}
                    elif request['action']=='chat':
                        if not session:raise RuntimeError('Load package first')
                        result={'ok':True,**session.chat(request['messages'],request['max_new_tokens'])}
                        result['fxbex_execution']=session.execution.last
                        result['memory']=session.memory
                    elif request['action']=='predict':
                        if not session:raise RuntimeError('Load package first')
                        result={'ok':True,**session.predict(request['inputs'])}
                        result['fxbex_execution']=session.execution.last
                        result['memory']=session.memory
                    elif request['action']=='generate':
                        if not session:raise RuntimeError('Load package first')
                        def token(text,token_ids):
                            print('@@FXBEX_TOKEN@@'+json.dumps({'text':text,'token_ids':token_ids},ensure_ascii=False),file=protocol,flush=True)
                        result={'ok':True,**session.generate(request['prompt'],request['max_tokens'],on_token=token)}
                        result['fxbex_execution']=session.execution.last
                        result['memory']=session.memory
                    elif request['action']=='unload':
                        memory=None
                        if session:
                            session.__exit__(None,None,None);memory=session.memory;session=None
                        result={'ok':True,'memory':memory}
                    else:raise ValueError('Unknown worker action')
            except Exception as exc:
                traceback.print_exc(file=sys.stderr);result={'ok':False,'error':str(exc)}
            print('@@FXBEX_RESPONSE@@'+json.dumps(result,ensure_ascii=False),file=protocol,flush=True)
            if request.get('action')=='unload':break
    finally:
        if session:session.__exit__(None,None,None)

if __name__=='__main__':main()
