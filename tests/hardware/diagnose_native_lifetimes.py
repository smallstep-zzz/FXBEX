"""Separate public SDK load phases from fxbX reload memory growth.

Each phase must run in a fresh process. No compilation or private SDK mutation.
Numerical checks use only the existing fixed [4,8] shared-linear fixture.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import weakref
import zipfile


def sample():
    result = {}
    for filename in ('status', 'smaps_rollup'):
        for line in (Path('/proc/self') / filename).read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Anonymous', 'Threads'):
                fields = value.split()
                result[key] = int(fields[0]) * (1024 if fields[-1] == 'kB' else 1)
    result['fds'] = len(list(Path('/proc/self/fd').iterdir()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('deserialize', 'wrap', 'upload', 'load', 'reuse', 'reload'), required=True)
    p.add_argument('--fxb', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=32)
    p.add_argument('--calls', type=int, default=64)
    p.add_argument('--profiler-mode', choices=('default', 'none', 'warn'), default='default')
    p.add_argument('--entry', choices=('prefill', 'cascade'), default='prefill')
    a = p.parse_args()
    if a.rounds < 3 or a.calls < 1:
        p.error('at least3 rounds and positive calls required')
    a.out.mkdir(parents=True, exist_ok=False)
    os.environ.pop('FXBX_TRACE_STEPS', None)
    if a.profiler_mode == 'warn':
        os.environ['FURIOSA_TORCH_PROFILE'] = 'warn'
    result = {'status': 'running', 'phase': a.phase, 'samples': [], 'calls': 0,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'diagnostic, no profiler; not NPU execution qualification'}
    started = time.monotonic()
    try:
        import torch
        import furiosa.torch as sdk
        from furiosa.torch.config import get_profiler, try_set_profiler
        result['profiler_mode'] = a.profiler_mode
        result['initial_profiler_type'] = type(get_profiler()).__name__
        if a.profiler_mode == 'none':
            previous = try_set_profiler(None)
            result['previous_profiler_type'] = type(previous).__name__
        result['current_profiler_type'] = type(get_profiler()).__name__
        from fxbx import GraphFXB, SafeTensorWeights
        from fxbex.execution_fxb import read_execution_plan
        from fxbex.settings import npu_lock
        from furiosa.torch.config import set_cpu_fallback
        torch.set_num_threads(2)
        plan = read_execution_plan(a.fxb)
        with zipfile.ZipFile(a.fxb) as archive:
            edfs = [archive.read(k['member']) for k in plan['kernels'].values()]
        weights = SafeTensorWeights(a.weights)
        w, b = weights['core.weight'], weights['core.bias']
        x = torch.full((4, 8), .125, dtype=torch.bfloat16)
        delta = torch.nn.functional.linear(x, w, b)
        depth = len(plan['entries'][a.entry]['steps']) if a.entry == 'cascade' else 1
        result['samples'].append(sample())
        refs = []

        def execute(runtime):
            runtime.reset()
            expected = torch.zeros_like(x)
            for _ in range(a.calls):
                for _ in range(depth):
                    expected = expected + delta
                actual, = runtime.run(a.entry, (x,))
                torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                result['calls'] += 1

        if a.phase in ('deserialize', 'wrap', 'upload'):
            with npu_lock():
                sdk.set_fusion(8); sdk.set_mesh_kind(sdk.MeshKind.Single())
                set_cpu_fallback(False)
                for _ in range(a.rounds):
                    objects = [sdk.ir.Runnable.from_serialized_edf(raw) for raw in edfs]
                    if a.phase != 'deserialize':
                        objects = [sdk.Runnable(obj) for obj in objects]
                        refs.extend(weakref.ref(obj) for obj in objects)
                    if a.phase == 'upload':
                        objects = [obj.to('furiosa:0') for obj in objects]
                        sdk.current_stream(torch.device('furiosa:0')).synchronize()
                    objects.clear()
                    gc.collect()
                    result['samples'].append(sample())
        elif a.phase == 'reuse':
            with GraphFXB(a.fxb, weights=weights) as runtime:
                for _ in range(a.rounds):
                    execute(runtime)
                    gc.collect()
                    result['samples'].append(sample())
            del runtime
        else:
            for _ in range(a.rounds):
                with GraphFXB(a.fxb, weights=weights) as runtime:
                    if a.phase == 'reload':
                        execute(runtime)
                del runtime
                gc.collect()
                result['samples'].append(sample())
        gc.collect()
        result['final'] = sample()
        result['live_python_wrappers'] = sum(ref() is not None for ref in refs)
        post = result['samples'][3:]
        result.update(status='completed', rss_growth_after_two_rounds=post[-1]['VmRSS'] - post[0]['VmRSS'],
                      private_growth_after_two_rounds=(post[-1]['Private_Clean'] + post[-1]['Private_Dirty']) - (post[0]['Private_Clean'] + post[0]['Private_Dirty']))
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}), flush=True)
    raise SystemExit(0 if result['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
