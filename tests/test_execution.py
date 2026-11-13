import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fxbex.execution import ExecutionRecorder, MemoryTelemetry, manifest_identity, memory_text
from fxbex.runtime import Session

class ExecutionTests(unittest.TestCase):
    def test_generate_is_recorded_and_runtime_failure_blocks_reuse(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            session=Session('unused',report_dir=d)
            session.manifest={'runtime':{'adapter':'qwen35_replay'}}
            session.adapter=Mock()
            session.execution=ExecutionRecorder(d,session.manifest)
            with patch('fxbex.hug_engine.generate',return_value={'text':'secret'}):
                session.generate('private',2)
            self.assertEqual(session.execution.last['operation'],'generate')
            self.assertNotIn('secret',session.execution.path.read_text())
            with self.assertRaises(ValueError):session.generate('private',-1)
            self.assertFalse(session.failed)
            session.adapter.predict.side_effect=ValueError('ambiguous adapter failure')
            with self.assertRaisesRegex(ValueError,'ambiguous'):session.predict({})
            self.assertTrue(session.failed)
            with self.assertRaisesRegex(RuntimeError,'unload and reload'):session.generate('a',1)

    def test_logical_peak_is_distinct_from_physical_usage(self):
        value = memory_text({'model_logical_memory': {'execution': {
            'peak_live_tensor_logical_bytes': 24576, 'released_tensor_bindings': 33}}})
        self.assertIn('SMI device DRAM: unavailable', value)
        self.assertIn('live tensor peak=0.02 MiB', value)
        self.assertIn('released bindings=33', value)

    def test_device_counter_supports_both_smi_formats(self):
        for memory in ({'used_size': 0}, {'DRAM': {'used_size': 0}}):
            self.assertIn('npu2=0.00 MiB (unconfirmed)', memory_text({
                'devices': [{'device': 'npu2', 'memory': memory}],
            }))

    def test_memory_telemetry_records_public_smi_json(self):
        with tempfile.TemporaryDirectory() as d:
            completed=type('Completed',(),{'stdout':'[{"memory":{"DRAM":{"used_size":7}}}]'})()
            with patch('fxbex.execution.subprocess.run',return_value=completed) as run:
                telemetry=MemoryTelemetry(d)
                row=telemetry.sample('after_load')
            run.assert_called_once_with(['furiosa-smi','status','--format','json'],capture_output=True,text=True,timeout=5,check=True)
            self.assertEqual(row['phase'],'after_load')
            self.assertEqual(row['devices'][0]['memory']['DRAM']['used_size'],7)
            self.assertEqual(json.loads((Path(d)/'memory.jsonl').read_text())['source'],'furiosa-smi status --format json')

    def test_missing_smi_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as d, patch('fxbex.execution.subprocess.run',side_effect=FileNotFoundError):
            row=MemoryTelemetry(d).sample('after_load')
            self.assertIsNone(row['physical_allocated_bytes'])
            self.assertEqual(row['error_type'],'FileNotFoundError')
            self.assertNotIn('devices',row)

    def test_identity_and_receipt_do_not_promote_old_trace(self):
        with tempfile.TemporaryDirectory() as d:
            recorder = ExecutionRecorder(d, {'model': 'x', 'files': {'a': 'hash'}})
            result = {'messages': ['private'], 'npu_tasks': None,
                      'current_request_traced': False, 'validated_audit': '/old/summary.json'}
            self.assertIs(recorder.call('chat', lambda: result), result)
            row = json.loads((Path(d)/'executions.jsonl').read_text())
            self.assertEqual(row['npu_verdict'], 'unknown')
            self.assertEqual(row['status'], 'succeeded')
            self.assertNotIn('private', json.dumps(row))
            self.assertFalse(row['adapter_reported']['current_request_traced'])
            self.assertEqual(manifest_identity({'a':1,'b':2}), manifest_identity({'b':2,'a':1}))
            self.assertNotEqual(manifest_identity({'a':1}), manifest_identity({'a':2}))

    def test_failed_execution_is_recorded_without_exception_content(self):
        with tempfile.TemporaryDirectory() as d:
            recorder=ExecutionRecorder(d,{})
            def fail(): raise ValueError('private prompt')
            with self.assertRaisesRegex(ValueError,'private prompt'):
                recorder.call('predict',fail)
            self.assertEqual(recorder.last['status'],'failed')
            self.assertEqual(recorder.last['error_type'],'ValueError')
            self.assertNotIn('private',Path(recorder.path).read_text())

    def test_session_routes_calls_and_cleanup_survives_reset_failure(self):
        with tempfile.TemporaryDirectory() as d:
            from unittest.mock import Mock
            session=Session('unused',report_dir=d)
            session.execution=ExecutionRecorder(d,{})
            session.adapter=Mock();session.adapter.predict.return_value={'outputs':[]}
            self.assertEqual(session.predict({}),{'outputs':[]})
            self.assertEqual(session.execution.last['operation'],'predict')
            released=[];session.stack.callback(lambda:released.append(True))
            torch=Mock();torch._dynamo.reset.side_effect=RuntimeError('reset')
            with patch.dict('sys.modules',{'torch':torch}):
                with self.assertRaisesRegex(RuntimeError,'reset'):session.__exit__()
            self.assertEqual(released,[True])

    def test_recording_failure_preserves_model_error(self):
        with tempfile.TemporaryDirectory() as d:
            recorder=ExecutionRecorder(d,{})
            def fail(): raise ValueError('model failure')
            with patch('pathlib.Path.open',side_effect=OSError('full disk')):
                with self.assertRaisesRegex(ValueError,'model failure'):
                    recorder.call('predict',fail)
            self.assertFalse(recorder.last['persisted'])

class TelemetryRateTests(unittest.TestCase):
    def test_request_snapshots_reuse_but_boundaries_and_force_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            completed=type('Completed',(),{'stdout':'[]'})()
            with patch('fxbex.execution.subprocess.run',return_value=completed) as command:
                telemetry=MemoryTelemetry(d)
                telemetry.sample('after_load')
                row=telemetry.sample('before_predict');telemetry.sample('after_predict')
                self.assertEqual(command.call_count,1)
                self.assertTrue(row['cached_snapshot']);self.assertGreaterEqual(row['sample_age_s'],0)
                telemetry.sample('after_predict',force=True);telemetry.sample('after_unload')
                self.assertEqual(command.call_count,3)
