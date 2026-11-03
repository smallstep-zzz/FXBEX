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
                        result={'ok':True,'metrics':getattr(session.adapter,'acceptance',{}),'report_dir':str(session.report_dir)}
                    elif request['action']=='chat':
                        if not session:raise RuntimeError('Load package first')
                        result={'ok':True,**session.chat(request['messages'],request['max_new_tokens'])}
                    elif request['action']=='predict':
                        if not session:raise RuntimeError('Load package first')
                        result={'ok':True,**session.predict(request['inputs'])}
                    elif request['action']=='unload':result={'ok':True}
                    else:raise ValueError('Unknown worker action')
            except Exception as exc:
                traceback.print_exc(file=sys.stderr);result={'ok':False,'error':str(exc)}
            print('@@FXBEX_RESPONSE@@'+json.dumps(result,ensure_ascii=False),file=protocol,flush=True)
            if request.get('action')=='unload':break
    finally:
        if session:session.__exit__(None,None,None)

if __name__=='__main__':main()
