"""Small execution receipts, not a replacement for official NPU trace verification."""
import hashlib
import json
import time
from pathlib import Path
import subprocess

def memory_text(snapshot):
    snapshot=snapshot or {}
    logical=snapshot.get('model_logical_memory',{})
    values=[]
    for key,label in [('weight_logical_bytes','weights'),('kv_logical_bytes','KV'),('recurrent_logical_bytes','recurrent')]:
        if key in logical:values.append(f"{label}={logical[key]/1024**3:.3f} GiB")
    execution=logical.get('execution') or {}
    if 'peak_live_tensor_logical_bytes' in execution:
        values.append(f"live tensor peak={execution['peak_live_tensor_logical_bytes']/1024**2:.2f} MiB")
        values.append(f"released bindings={execution.get('released_tensor_bindings',0)}")
    return 'physical HBM unmeasured; SMI zero is inconclusive. Logical NPU tensors: '+(', '.join(values) or 'not reported')



class MemoryTelemetry:
    """Best-effort SMI memory snapshots; never an allocator substitute."""
    def __init__(self, report_dir, interval_s=5):
        self.path = Path(report_dir) / 'memory.jsonl'
        self.last = None
        self.interval_s = interval_s
        self.last_sample_monotonic = None

    def sample(self, phase, *, force=False):
        now = time.monotonic()
        boundary = phase in ('before_load', 'after_load', 'after_unload')
        if (self.last is not None and not boundary and not force
                and now-self.last_sample_monotonic < self.interval_s):
            return dict(self.last, requested_phase=phase, cached_snapshot=True,
                        sample_age_s=now-self.last_sample_monotonic)
        row = {'schema': 1, 'phase': phase, 'time_unix': time.time(),
               'source': 'furiosa-smi status --format json',
               'physical_allocated_bytes':None,
               'note':'SMI-reported counters are not allocator proof; zero does not establish zero HBM use.'}
        try:
            result = subprocess.run(['furiosa-smi', 'status', '--format', 'json'],
                                    capture_output=True, text=True, timeout=5, check=True)
            row['devices'] = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            row['error_type'] = type(exc).__name__
        self.last = row
        self.last_sample_monotonic = now
        try:
            with self.path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        except (OSError,ValueError,TypeError):
            # A diagnostic must never hide model execution or cleanup failures.
            pass
        return row


def manifest_identity(manifest):
    # The validated manifest already contains hashes of packaged artifacts.
    # Hashing it avoids rereading multi-GB weights for every inference.
    data = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


class ExecutionRecorder:
    def __init__(self, report_dir, manifest):
        self.path = Path(report_dir) / 'executions.jsonl'
        self.identity = manifest_identity(manifest)
        self.sequence = 0
        self.last = None
        self.backend=manifest.get('optimization',{}).get('backend',
            'torch_runnable' if manifest.get('runtime',{}).get('adapter') in ('qwen35_replay','runnable_dag') else 'unknown')

    def call(self, operation, callback):
        self.sequence += 1
        started = time.perf_counter()
        row = {'schema': 1, 'sequence': self.sequence, 'operation': operation,
               'manifest_sha256': self.identity, 'status': 'failed',
               'timing_scope': 'host call including dispatch/transfers/waits; completion is adapter-defined',
               'npu_verdict': 'unknown', 'support_status': 'not_established',
               'optimization_status': 'not_measured', 'adapter_reported': {}}
        row.update(backend=self.backend,measurement_scope='adapter_defined_return',
                   unit='seconds',clock_domain='host_monotonic',completion_confirmed=None,
                   profiling_enabled=None)
        try:
            result = callback()
            row['status'] = 'succeeded'
            if isinstance(result, dict):
                # Never save prompts, messages, tensor outputs or arbitrary adapter data.
                for key in ('compiler_calls', 'npu_tasks', 'current_request_traced'):
                    value = result.get(key)
                    if type(value) in (int, bool) or value is None:
                        row['adapter_reported'][key] = value
                for key in ('validated_audit',):
                    value = result.get(key)
                    if isinstance(value, str) and len(value) <= 4096:
                        row['adapter_reported'][key] = value
            return result
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['wall_seconds'] = time.perf_counter() - started
            row['report_path'] = str(self.path)
            self.last = row
            row['persisted'] = True
            try:
                with self.path.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            except OSError as exc:
                # Telemetry failure must not hide the original model exception.
                row['persisted'] = False
                row['recording_error_type'] = type(exc).__name__
