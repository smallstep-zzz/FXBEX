"""Bound memory of a multi-model native campaign with one process per job.

Reuses diagnose_native_lifetimes.py and a qualified shared-linear EDF fixture.
The parent imports neither Torch nor the SDK. This validates process containment
and numerical replay, not per-call NPU evidence (collected separately).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('RSS unavailable')


def npu_memory():
    rows = json.loads(subprocess.check_output(['furiosa-smi', 'status', '--format', 'json'], text=True, timeout=15))
    return {r['device']: r['memory']['DRAM']['used_size'] for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=32)
    p.add_argument('--calls', type=int, default=16, help='calls per child round; four rounds per job')
    p.add_argument('--entry', choices=('prefill', 'cascade'), default='cascade')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--max-parent-growth-mib', type=int, default=128)
    a = p.parse_args()
    if a.jobs < 3 or a.calls < 1 or a.timeout <= 0 or a.max_parent_growth_mib < 0:
        p.error('at least3 jobs, positive calls/timeout and nonnegative bound required')
    a.out.mkdir(parents=True, exist_ok=False)
    from fxbex.settings import npu_lock, lock_path
    result = {'status': 'running', 'jobs': [], 'calls': 0,
              'scope': 'fresh native process per model job; reused synthetic fixed-shape model',
              'npu_evidence': 'not_collected_here; require separate strict qualification',
              'collection': 'FURIOSA_TORCH_PROFILE=warn before child SDK import'}
    started = time.monotonic()
    try:
        with npu_lock() as lease:
            baseline = npu_memory()
            if any(baseline.values()):
                raise RuntimeError('infrastructure_busy: device memory is already occupied')
            env = dict(os.environ, FURIOSA_TORCH_PROFILE='warn',
                       FXBEX_NPU_LOCK=str(lock_path()), FXBEX_NPU_LOCK_FD=str(lease))
            env.pop('FXBX_TRACE_STEPS', None)
            result['parent_rss_before'] = rss()
            for i in range(a.jobs):
                out = a.out / f'job-{i:03d}'
                command = [sys.executable, str(Path(__file__).with_name('diagnose_native_lifetimes.py')),
                           '--phase', 'reuse', '--entry', a.entry, '--profiler-mode', 'warn',
                           '--fxb', str(a.fxb), '--weights', str(a.weights), '--out', str(out),
                           '--rounds', '4', '--calls', str(a.calls)]
                tick = time.monotonic()
                with (a.out / f'job-{i:03d}.log').open('wb') as log:
                    child = subprocess.run(command, env=env, pass_fds=(lease,), stdout=log,
                                           stderr=subprocess.STDOUT, timeout=a.timeout)
                row = json.loads((out / 'result.json').read_text())
                if child.returncode or row['status'] != 'completed':
                    raise RuntimeError(f'child job {i} failed; see preserved log/result')
                memory = npu_memory()
                if memory != baseline:
                    raise RuntimeError(f'child job {i} left unexpected NPU memory: {memory}')
                result['jobs'].append({'index': i, 'seconds': time.monotonic() - tick,
                                       'parent_rss': rss(), 'worker_final_rss': row['final']['VmRSS'],
                                       'calls': row['calls'], 'worker_reaped': True, 'closed_npu_bytes': memory})
                result['calls'] += row['calls']
                (a.out / 'progress.json').write_text(json.dumps({'jobs': i + 1, 'calls': result['calls']}))
            if any(n == 'torch' or n.startswith('furiosa.torch') for n in sys.modules):
                raise AssertionError('parent imported the model SDK')
            post = result['jobs'][1:]
            growth = max(r['parent_rss'] for r in post) - post[0]['parent_rss']
            if growth > a.max_parent_growth_mib * 1024**2:
                raise AssertionError('parent memory growth exceeded explicit bound')
            result.update(status='passed', parent_rss_growth_after_warmup=growth,
                          memory_outcome='contained_by_worker_lifecycle',
                          in_process_sdk_retention='not_fixed_by_this_probe',
                          parent_imported_sdk=False, growth_bound_mib=a.max_parent_growth_mib)
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'jobs'}), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
