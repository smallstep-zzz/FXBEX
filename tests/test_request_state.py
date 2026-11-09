"""Request isolation/ownership; fake SDK tests are not NPU evidence."""
from contextlib import nullcontext
from unittest.mock import patch
import unittest
import test_execution_fxb as fixtures
from fxbex.execution_fxb import ExecutionFXB,ExecutionPlanError,RequestState

class RequestStateTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ExecutionPlanTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
    def test_interleaved_clients_reset_release_and_legacy_state_are_isolated(self):
        f=self.fixture;f.export();modules,_=f.backend()
        with patch.dict('sys.modules',modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(f.output,max_request_states=2) as model:
                with model.new_state() as a,model.new_state() as b:
                    self.assertEqual(model.predict((fixtures.Tensor(1),),state=a)[0].value,2)
                    self.assertEqual(model.predict((fixtures.Tensor(10),),state=b)[0].value,20)
                    self.assertEqual(model.predict((fixtures.Tensor(1),),state=a)[0].value,6)
                    self.assertFalse(model.states)  # No implicit default cache beside clients.
                    model.reset(state=a)
                    self.assertEqual(model.predict((fixtures.Tensor(1),),state=a)[0].value,2)
                    self.assertEqual(model.predict((fixtures.Tensor(10),),state=b)[0].value,60)
                    self.assertGreaterEqual(a.last_timing['queue_wait_s'],0)
                    self.assertGreater(a.last_timing['request_s'],0)
                with self.assertRaisesRegex(ExecutionPlanError,'closed session'):model.predict((fixtures.Tensor(1),),state=a)
                self.assertEqual(model.predict((fixtures.Tensor(1),))[0].value,2)
                self.assertFalse(model.failed)
    def test_foreign_and_borrowed_handles_never_launch(self):
        f=self.fixture;f.export();modules,calls=f.backend()
        with patch.dict('sys.modules',modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(f.output) as model:
                with self.assertRaisesRegex(ExecutionPlanError,'another'):model.predict((fixtures.Tensor(1),),state=RequestState(object(),{}))
                state=model.new_state();state._borrowed=True
                with self.assertRaisesRegex(ExecutionPlanError,'borrowed'):model.predict((fixtures.Tensor(1),),state=state)
                state._borrowed=False;model.release_state(state)
                self.assertFalse(any(c[0]=='run' for c in calls));self.assertFalse(model.failed)
    def test_admission_counts_extra_clients_before_allocating(self):
        f=self.fixture;f.plan['memory_policy']={'budget_bytes':31};f.export();modules,_=f.backend()
        with patch.dict('sys.modules',modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(f.output,max_request_states=2) as model:
                state=model.new_state()
                with self.assertRaisesRegex(ExecutionPlanError,'request states'):model.new_state()
                self.assertFalse(model.failed);self.assertEqual(len(model._requests),1)
                self.assertEqual(model.predict((fixtures.Tensor(1),),state=state)[0].value,2)
    def test_default_request_limit_prevents_unbounded_allocations(self):
        f=self.fixture;f.export();modules,_=f.backend()
        with patch.dict('sys.modules',modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(f.output) as model:
                model.new_state()
                with self.assertRaisesRegex(ExecutionPlanError,'limit reached'):model.new_state()
                self.assertFalse(model.failed)
    def test_model_close_invalidates_outstanding_handles(self):
        f=self.fixture;f.export();modules,_=f.backend()
        with patch.dict('sys.modules',modules),patch('importlib.metadata.version',return_value='test'),patch('fxbex.settings.npu_lock',return_value=nullcontext()):
            with ExecutionFXB(f.output) as model:state=model.new_state()
            self.assertTrue(state._closed);self.assertFalse(state._tensors);state.close()

if __name__=='__main__':unittest.main()
