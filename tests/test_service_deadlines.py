"""Real pipe/process controls without importing Torch or touching an NPU."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fxbex.service import PackageService,WorkerCancelled,WorkerTimeout,bounded_lock


class WorkerControl(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.service=PackageService(self.root/'fixture.fxbex')
        self.service.timeouts.update(request=.3,load=.3,unload=.3,terminate=.1,kill=.5)
        self.addCleanup(self.service.stop)

    def start(self,code):
        service=self.service
        service.log_path=self.root/'worker.log';service.log=service.log_path.open('w')
        service.process=subprocess.Popen([sys.executable,'-u','-c',code],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=service.log,bufsize=0)
        os.set_blocking(service.process.stdin.fileno(),False)
        os.set_blocking(service.process.stdout.fileno(),False)
        service.loaded=True
        return service.process

    def test_silent_and_partial_line_cannot_wait_forever(self):
        for prefix in ('',"sys.stdout.write('@@FXBEX_RESPONSE@@{');sys.stdout.flush();"):
            with self.subTest(prefix=prefix):
                process=self.start('import sys,time;sys.stdin.readline();'+prefix+'time.sleep(30)')
                started=time.monotonic()
                with self.assertRaises(WorkerTimeout):self.service.predict({})
                self.assertLess(time.monotonic()-started,3)
                self.assertIsNotNone(process.poll())
                self.assertIsNone(self.service.process)
                self.assertFalse(self.service.loaded)
                self.assertTrue(self.service.last_failure['worker_reaped'])

    def test_progress_does_not_extend_deadline(self):
        self.start("import sys,time\nsys.stdin.readline()\nwhile True:\n print('@@FXBEX_PROGRESS@@{\"phase\":\"busy\",\"label\":\"working\"}',flush=True)\n time.sleep(.01)")
        events=[];started=time.monotonic()
        with self.assertRaises(WorkerTimeout):self.service.chat([{'role':'user','content':'x'}],on_progress=lambda *event:events.append(event))
        self.assertLess(time.monotonic()-started,3)
        self.assertTrue(events)

    def test_stdin_backpressure_is_bounded(self):
        self.start('import time;time.sleep(30)')
        with self.assertRaises(WorkerTimeout):self.service.predict({'value':'a'*(3*1024**2)})
        self.assertEqual(self.service.last_failure['stage'],'request_write')

    def test_partial_utf8_tokens_and_response(self):
        code="""import sys,time
sys.stdin.readline()
raw='@@FXBEX_TOKEN@@{"text":"한","token_ids":[1]}\\n@@FXBEX_RESPONSE@@{"ok":true,"text":"한"}\\n'.encode()
for value in raw:
 sys.stdout.buffer.write(bytes([value]));sys.stdout.buffer.flush()
"""
        self.start(code);events=[]
        actual=self.service.generate('hello',max_tokens=2,on_token=events.append)
        self.assertEqual(actual['text'],'한')
        self.assertEqual(events,[{'text':'한','token_ids':[1]}])

    def test_response_before_deadline_succeeds(self):
        self.start("import sys,time;sys.stdin.readline();time.sleep(.05);print('@@FXBEX_RESPONSE@@{\"ok\":true}',flush=True)")
        self.assertTrue(self.service.predict({})['ok'])

    def test_eof_before_response_is_failure(self):
        self.start('import sys;sys.stdin.readline()')
        with self.assertRaisesRegex(RuntimeError,'stdout ended'):self.service.predict({})
        self.assertIsNone(self.service.process)

    def test_oversized_unterminated_frame_is_failure(self):
        self.start("import sys;sys.stdin.readline();sys.stdout.buffer.write(b'x'*(4*1024**2+65536));sys.stdout.buffer.flush()")
        with self.assertRaisesRegex(RuntimeError,'frame exceeds limit'):self.service.predict({})

    def test_oversized_outgoing_request_is_rejected_without_stopping_worker(self):
        process=self.start('import time;time.sleep(30)')
        with self.assertRaisesRegex(ValueError,'request exceeds transport frame limit'):
            self.service.predict({'value':'x'*(5*1024**2)})
        self.assertTrue(self.service.loaded)
        self.assertIs(self.service.process,process)
        self.assertIsNone(process.poll())

    def test_queue_timeout_does_not_stop_active_worker(self):
        process=self.start('import time;time.sleep(30)')
        acquired=threading.Event();release=threading.Event()
        def owner():
            with self.service.lock:acquired.set();release.wait(3)
        thread=threading.Thread(target=owner);thread.start();self.addCleanup(thread.join,3)
        self.assertTrue(acquired.wait(2))
        try:
            with self.assertRaises(WorkerTimeout) as error:self.service.predict({},deadline=time.monotonic()+.05)
            self.assertEqual(error.exception.code,'queue_timeout')
            self.assertIsNone(process.poll())
            self.assertIs(self.service.process,process)
        finally:release.set();thread.join(2)

    def test_cancelled_queue_does_not_stop_active_worker(self):
        process=self.start('import time;time.sleep(30)')
        event=threading.Event();event.set()
        with self.assertRaises(WorkerCancelled):self.service.predict({},cancel_event=event)
        self.assertIs(self.service.process,process)
        self.assertIsNone(process.poll())
        self.assertTrue(self.service.loaded)

    def test_cancel_event_stops_only_its_active_request(self):
        process=self.start('import sys,time;sys.stdin.readline();time.sleep(30)')
        event=threading.Event();errors=[];started=threading.Event()
        original=self.service._request
        def observe(*args,**kwargs):started.set();return original(*args,**kwargs)
        def request():
            try:self.service.predict({},deadline=time.monotonic()+10,cancel_event=event)
            except BaseException as error:errors.append(error)
        with patch.object(self.service,'_request',side_effect=observe):
            thread=threading.Thread(target=request);thread.start();self.assertTrue(started.wait(2))
            event.set();thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0],WorkerCancelled)
        self.assertIsNotNone(process.poll())

    def test_cancel_does_not_need_request_lock(self):
        self.start('import sys,time;sys.stdin.readline();time.sleep(30)')
        errors=[];active=threading.Event()
        original=self.service._request
        def request(*args,**kwargs):active.set();return original(*args,**kwargs)
        with patch.object(self.service,'_request',side_effect=request):
            thread=threading.Thread(target=lambda:self.capture_error(errors))
            thread.start();self.assertTrue(active.wait(2))
            self.assertTrue(self.service.cancel());thread.join(3)
        self.assertFalse(thread.is_alive());self.assertIsInstance(errors[0],WorkerCancelled)

    def capture_error(self,errors):
        try:self.service.predict({},deadline=time.monotonic()+10)
        except BaseException as error:errors.append(error)

    def test_terminate_ignored_escalates_to_kill(self):
        process=self.start("import signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);sys.stdin.readline();time.sleep(30)")
        with self.assertRaises(WorkerTimeout):self.service.predict({})
        self.assertIsNotNone(process.poll())
        self.assertLess(process.returncode,0)

    def test_cleanup_error_preserves_timeout(self):
        self.start('import time;time.sleep(30)')
        with patch.object(self.service,'_stop_process',side_effect=RuntimeError('cleanup failed')):
            with self.assertRaises(WorkerTimeout) as error:self.service.predict({})
        self.assertEqual(str(error.exception.__cause__),'cleanup failed')

    def test_callback_cannot_interleave_a_reentrant_request(self):
        self.start("import sys,time;sys.stdin.readline();print('@@FXBEX_PROGRESS@@{\"phase\":\"busy\",\"label\":\"working\"}',flush=True);time.sleep(30)")
        with self.assertRaisesRegex(RuntimeError,'worker_reentrant_request'):
            self.service.chat([{'role':'user','content':'x'}],on_progress=lambda *args:self.service.predict({}))
        self.assertIsNone(self.service.process)
        self.assertIsNone(self.service._active)

    def test_unreaped_worker_handle_is_retained(self):
        class Unreaped:
            stdin=io.BytesIO();stdout=io.BytesIO()
            def poll(self):return None
            def terminate(self):pass
            def kill(self):pass
            def wait(self,timeout=None):raise subprocess.TimeoutExpired('dummy',timeout)
        self.service.process=process=Unreaped()
        try:
            with self.assertRaisesRegex(RuntimeError,'worker_unreaped'):self.service.stop()
            self.assertIs(self.service.process,process)
            self.assertTrue(self.service.failed_unreaped)
            with self.assertRaisesRegex(RuntimeError,'worker_unreaped'):self.service._start()
        finally:self.service.process=None;self.service.failed_unreaped=False

    def test_stop_between_response_and_load_commit_cannot_revive_loaded(self):
        self.service.timeouts['load']=5
        process=self.start("import sys,time;sys.stdin.readline();print('@@FXBEX_RESPONSE@@{\"ok\":true}',flush=True);time.sleep(30)")
        self.service.loaded=False
        entered=threading.Event();release=threading.Event();errors=[]
        class PausedResponse(dict):
            def get(self,key,*default):
                if key=='context_limit':entered.set();release.wait(3)
                return super().get(key,*default)
        def request(*args,**kwargs):return PausedResponse(ok=True)
        def load():
            try:self.service.load()
            except BaseException as error:errors.append(error)
        with patch.object(self.service,'_start'),patch.object(self.service,'_stop_process') as cleanup,patch.object(self.service,'_request',side_effect=request):
            # The old worker is a stand-in for the child the patched _start owns.
            real_stop=PackageService._stop_process.__get__(self.service)
            cleanup.side_effect=lambda:None
            thread=threading.Thread(target=load);thread.start();self.assertTrue(entered.wait(2))
            try:
                cleanup.side_effect=real_stop
                self.service.stop()
            finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.service.loaded)
        self.assertIsNone(self.service.process)
        self.assertIsNotNone(process.poll())
        self.assertIsInstance(errors[0],WorkerCancelled)

    def test_stop_waits_for_spawn_registration_and_reaps_that_child(self):
        self.service.timeouts['load']=5
        real_popen=subprocess.Popen;created=threading.Event();release=threading.Event()
        stop_entered=threading.Event();stop_done=threading.Event();children=[];errors=[]
        code="import sys,time;sys.stdin.readline();print('@@FXBEX_RESPONSE@@{\"ok\":true}',flush=True);time.sleep(30)"
        def delayed_spawn(args,**kwargs):
            child=real_popen([sys.executable,'-u','-c',code],**kwargs);children.append(child)
            created.set();release.wait(3)
            return child
        def load():
            try:self.service.load()
            except BaseException as error:errors.append(error)
        def stop():
            stop_entered.set()
            try:self.service.stop()
            except BaseException as error:errors.append(error)
            finally:stop_done.set()
        with patch('fxbex.service.runs_path',return_value=self.root/'runs'),patch('fxbex.service.subprocess.Popen',side_effect=delayed_spawn):
            loader=threading.Thread(target=load);loader.start();self.assertTrue(created.wait(2))
            stopper=threading.Thread(target=stop);stopper.start();self.assertTrue(stop_entered.wait(2))
            try:self.assertFalse(stop_done.wait(.1),'stop returned before the new child was registered')
            finally:
                release.set();loader.join(3);stopper.join(3)
                for child in children:
                    if child.poll() is None:child.kill();child.wait(timeout=2)
        self.assertFalse(loader.is_alive() or stopper.is_alive())
        self.assertFalse(self.service.loaded)
        self.assertIsNone(self.service.process)
        self.assertTrue(children)
        self.assertTrue(all(child.poll() is not None for child in children))


if __name__=='__main__':unittest.main()
