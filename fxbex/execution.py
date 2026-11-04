"""Small execution receipts, not a replacement for official NPU trace verification."""
import hashlib
import json
import time
from pathlib import Path


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

    def call(self, operation, callback):
        self.sequence += 1
        started = time.perf_counter()
        row = {'schema': 1, 'sequence': self.sequence, 'operation': operation,
               'manifest_sha256': self.identity, 'status': 'failed',
               'timing_scope': 'host call including dispatch/transfers/waits; completion is adapter-defined',
               'npu_verdict': 'unknown', 'support_status': 'not_established',
               'optimization_status': 'not_measured', 'adapter_reported': {}}
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
