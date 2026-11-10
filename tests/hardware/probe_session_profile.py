"""Check one profiler lifetime around session loads, calls, completion and cleanup.

A separate diagnostic for SDK per-call profiler reference-time loss. It does not
change runtime or producer qualification policy. Every call has a unique enclosing
span; each required execution-plan span and NPU Task must fit inside that call.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reloads', type=int, default=4)
    p.add_argument('--calls', type=int, default=6)
    a = p.parse_args()
    if a.reloads < 1 or a.calls < 3:
        p.error('positive reloads and at least3 calls required')
    a.out.mkdir(parents=True, exist_ok=False)
    os.environ['FXBX_TRACE_STEPS'] = '1'
    import torch
    from fxbx import GraphFXB, SafeTensorWeights
    from fxbex.execution_fxb import read_execution_plan
    from furiosa.torch.profiler import Profiler, RecordFunction
    torch.set_num_threads(2)
    plan = read_execution_plan(a.fxb)
    weights = SafeTensorWeights(a.weights)
    weight, bias = weights['core.weight'], weights['core.bias']
    depth = len(plan['entries']['cascade']['steps'])
    result = {'status': 'running', 'strategy': 'single Profiler enclosing all session opens/calls/closes',
              'scope': 'bounded fixed-shape synthetic fixture; no SDK policy bypass'}
    started = time.monotonic(); calls = []
    try:
        trace = a.out / 'session-trace.json'
        with Profiler(str(trace)):
            for reload in range(a.reloads):
                with GraphFXB(a.fxb, weights=SafeTensorWeights(a.weights)) as runtime:
                    reference = torch.zeros(4, 8, dtype=torch.bfloat16)
                    for i in range(a.calls):
                        entry = ('prefill', 'decode', 'cascade')[i % 3]
                        x = torch.full((1, 8) if entry == 'decode' else (4, 8), .125 * (i - 2), dtype=torch.bfloat16)
                        with torch.no_grad():
                            delta = torch.nn.functional.linear(x, weight, bias).expand(4, 8).contiguous()
                            for _ in range(depth if entry == 'cascade' else 1):
                                reference = delta + reference
                        label = f'audit_call::{len(calls)}'
                        tick = time.monotonic()
                        with RecordFunction(label, propagate=True):
                            actual, = runtime.run(entry, (x,))
                        torch.testing.assert_close(actual, reference, atol=.02, rtol=.02)
                        calls.append({'label': label, 'entry': entry, 'reload': reload, 'host_seconds': time.monotonic() - tick})
        events = json.loads(trace.read_text())['traceEvents']
        cpu = sum('run_by_cpu' in e.get('name', '') for e in events)
        assert not cpu
        checks = []
        for call in calls:
            frames = [e for e in events if e.get('name') == call['label'] and e.get('ph') == 'X']
            if len(frames) != 1:
                raise AssertionError(f"call span missing/ambiguous: {call['label']}")
            frame = frames[0]; low = frame['ts']; high = low + frame['dur']
            window = [e for e in events if 'ts' in e and low <= e['ts'] and e['ts'] + e.get('dur', 0) <= high]
            expected = {f"execution_plan::{call['entry']}::{i}::{s['kernel']}"
                        for i, s in enumerate(plan['entries'][call['entry']]['steps'])}
            spans = {e['name'] for e in window if e.get('name', '').startswith('execution_plan::')}
            tasks = {e['name'].split('::Task::', 1)[1] for e in window if '::Task::execution_plan::' in e.get('name', '')}
            if spans != expected or tasks != expected:
                raise AssertionError(f"{call['label']} evidence mismatch expected={len(expected)} spans={len(spans)} tasks={len(tasks)}")
            checks.append({**call, 'matched_steps': len(expected)})
        result.update(status='passed', calls=len(calls), reloads=a.reloads,
                      call_bounded_steps=sum(x['matched_steps'] for x in checks), cpu_routes=cpu, checks=checks)
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), completed_numerical_calls=len(calls))
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__': main()
