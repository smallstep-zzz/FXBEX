"""Bounded real-NPU stress of the shared-linear prefill/decode FXB fixture.

Requires an existing qualified program and weights from probe_checkpoint_weights.
No TCLMachine or compiler imports. Repeats genuine EDFs to test graph execution
and liveness independently of operator support. This is not a 9B model test.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def memory_sample(label):
    rss = None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            rss = int(line.split()[1]) * 1024
    devices = json.loads(subprocess.check_output(['furiosa-smi', 'status', '--format', 'json'], text=True))
    return {'label': label, 'rss_bytes': rss,
            'npu_bytes': {d['device']: d['memory']['DRAM']['used_size'] for d in devices}}


def cascade_package(source, output, depth):
    from fxbex.execution_fxb import read_execution_plan, export_execution_plan
    plan = read_execution_plan(source)
    step = plan['entries']['prefill']['steps']
    if len(step) != 1 or plan['entries']['prefill']['state_updates'].keys() != {'state:s'}:
        raise ValueError('This stress fixture needs the shared-linear prefill/decode program')
    step = step[0]
    if len(step['outputs']) != 1:
        raise ValueError('Expected one prefill output')
    original_output = step['outputs'][0]
    previous = 'state:s'; steps = []
    for i in range(depth):
        name = f'cascade:{i}'
        plan['tensors'][name] = copy.deepcopy(plan['tensors'][original_output])
        steps.append({'kernel': step['kernel'],
                      'inputs': [previous if n == 'state:s' else n for n in step['inputs']],
                      'outputs': [name]})
        previous = name
    plan['entries']['cascade'] = {'inputs': plan['entries']['prefill']['inputs'],
                                 'steps': steps, 'outputs': [previous],
                                 'state_updates': {'state:s': previous}}
    plan['source_metadata'] = {'architecture': 'SyntheticRepeatedQualifiedLinear',
                               'scope': 'runtime graph and liveness stress; not a larger pretrained model',
                               'source_fxb_sha256': digest(source), 'cascade_depth': depth}
    directory = output.parent / 'edfs'; directory.mkdir()
    with zipfile.ZipFile(source) as archive:
        for i, kernel in enumerate(plan['kernels'].values()):
            path = directory / f'{i}.edf'
            path.write_bytes(archive.read(kernel.pop('member')))
            kernel['edf_path'] = str(path)
    export_execution_plan(plan, output)
    return read_execution_plan(output)


def assert_trace(trace, plan, entry):
    events = json.loads(trace.read_text())['traceEvents']
    expected = {f"execution_plan::{entry}::{i}::{step['kernel']}"
                for i, step in enumerate(plan['entries'][entry]['steps'])}
    spans = {e['name'] for e in events if e.get('name', '').startswith('execution_plan::')}
    tasks = {e['name'].split('::Task::', 1)[1] for e in events
             if '::Task::execution_plan::' in e.get('name', '')}
    cpu = sum('run_by_cpu' in e.get('name', '') for e in events)
    if spans != expected or tasks != expected or cpu:
        raise AssertionError(f'trace mismatch: expected={len(expected)}, spans={len(spans)}, tasks={len(tasks)}, cpu={cpu}')
    return {'entry': entry, 'expected_steps': len(expected), 'npu_steps': len(tasks),
            'cpu_routes': cpu, 'trace_sha256': digest(trace)}


def run(args):
    import torch
    from fxbx import GraphFXB, SafeTensorWeights
    from fxbex.execution_fxb import ExecutionPlanError
    from furiosa.torch.profiler import Profiler
    os.environ['FXBX_TRACE_STEPS'] = '1'
    torch.set_num_threads(2)
    root = args.out; root.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scope': 'synthetic fixed [4,8] shared-linear program; no large-checkpoint or long-service claim',
              'source_fxb_sha256': digest(args.fxb), 'calls': 0, 'reloads': [], 'memory': [], 'traces': [], 'trace_failures': [],
              'continue_on_trace_loss': args.continue_on_trace_loss}
    started = time.monotonic(); samples = []; total_rejections = 0
    try:
        plan = cascade_package(args.fxb, root / 'cascade.fxb', args.depth)
        weights = SafeTensorWeights(args.weights)
        weight, bias = weights['core.weight'], weights['core.bias']
        result['cascade_depth'] = args.depth
        result['plan_steps'] = {k: len(v['steps']) for k, v in plan['entries'].items()}
        result['unique_edfs'] = len(plan['kernels'])
        result['memory'].append(memory_sample('before_load'))
        for reload in range(args.reloads):
            load_started = time.monotonic()
            with GraphFXB(root / 'cascade.fxb', weights=SafeTensorWeights(args.weights), max_request_states=args.states) as runtime:
                load_s = time.monotonic() - load_started
                states = [runtime.new_state() for _ in range(args.states)]
                references = [torch.zeros(4, 8, dtype=torch.bfloat16) for _ in states]
                # Admission failure must not allocate/poison an otherwise healthy session.
                try:
                    runtime.new_state()
                except ExecutionPlanError:
                    total_rejections += 1
                else:
                    raise AssertionError('state count limit was ignored')
                try:
                    runtime.run('decode', (torch.ones(2, 8, dtype=torch.bfloat16),), state=states[0])
                except ExecutionPlanError:
                    total_rejections += 1
                else:
                    raise AssertionError('wrong input shape was accepted')
                closed = states.pop(); closed.close()
                try:
                    runtime.run('decode', (torch.ones(1, 8, dtype=torch.bfloat16),), state=closed)
                except ExecutionPlanError:
                    total_rejections += 1
                else:
                    raise AssertionError('closed request state was accepted')
                states.append(runtime.new_state())
                result['memory'].append(memory_sample(f'loaded:{reload}'))
                profiled = set(); latencies = []
                for i in range(args.calls):
                    index = i % len(states)
                    if i % 16 == 0:
                        runtime.reset(state=states[index]); references[index].zero_()
                    entry = ('prefill', 'decode', 'cascade')[i % 3]
                    value = .25 * ((i % 5) - 2)
                    shape = (1, 8) if entry == 'decode' else (4, 8)
                    x = torch.full(shape, value, dtype=torch.bfloat16)
                    with torch.no_grad():
                        delta = torch.nn.functional.linear(x, weight, bias)
                        delta = delta.expand(4, 8).contiguous()
                        reference = references[index]
                        for _ in range(args.depth if entry == 'cascade' else 1):
                            reference = delta + reference
                    if entry not in profiled:
                        trace = root / f'trace-{reload}-{entry}.json'
                        with Profiler(str(trace)):
                            actual, = runtime.run(entry, (x,), state=states[index])
                        try:
                            result['traces'].append(assert_trace(trace, plan, entry))
                        except AssertionError as error:
                            result['trace_failures'].append({'reload': reload, 'entry': entry,
                                'trace': trace.name, 'trace_sha256': digest(trace), 'error': str(error)})
                            if not args.continue_on_trace_loss:
                                raise
                        profiled.add(entry)
                    else:
                        tick = time.monotonic()
                        actual, = runtime.run(entry, (x,), state=states[index])
                        latencies.append(time.monotonic() - tick)
                    torch.testing.assert_close(actual, reference, atol=.02, rtol=.02)
                    references[index] = reference
                    result['calls'] += 1
                    if i % 64 == 63:
                        result['memory'].append(memory_sample(f'loaded:{reload}:call:{i+1}'))
                for state in states:
                    state.close()
                samples.extend(latencies)
            result['reloads'].append({'index': reload, 'load_s': load_s, 'calls': args.calls,
                                      'warm_host_request_median_s': statistics.median(latencies)})
            result['memory'].append(memory_sample(f'closed:{reload}'))
            (root / 'progress.json').write_text(json.dumps({'reloads': reload + 1, 'calls': result['calls']}))
        closed = [m for m in result['memory'] if m['label'].startswith('closed:')]
        # Ignore first-load allocator warmup. Bounds are explicit and reported.
        rss_growth = max(m['rss_bytes'] for m in closed[1:]) - closed[1]['rss_bytes']
        devices = set(closed[1]['npu_bytes'])
        npu_growth = {d: max(m['npu_bytes'][d] for m in closed[1:]) - closed[1]['npu_bytes'][d] for d in devices}
        if rss_growth > args.max_growth_mib * 1024**2 or any(n > args.max_growth_mib * 1024**2 for n in npu_growth.values()):
            raise AssertionError('post-warmup memory growth exceeded the explicit bound')
        result.update(status='runtime_numerical_passed_evidence_incomplete' if result['trace_failures'] else 'passed', expected_calls=args.reloads * args.calls,
                      negative_requests_rejected=total_rejections, request_isolation='passed', reset='passed',
                      cpu_reference_comparisons=result['calls'], rss_growth_after_warmup_bytes=rss_growth,
                      npu_growth_after_warmup_bytes=npu_growth, growth_bound_mib=args.max_growth_mib,
                      warm_host_request_median_s=statistics.median(samples))
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (root / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('memory', 'traces')}), flush=True)
    return result['status'] == 'passed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, default=128)
    p.add_argument('--reloads', type=int, default=4)
    p.add_argument('--calls', type=int, default=128, help='calls per reload')
    p.add_argument('--states', type=int, default=4)
    p.add_argument('--max-growth-mib', type=int, default=128)
    p.add_argument('--continue-on-trace-loss', action='store_true', help='finish numeric/memory checks but return failure if any sampled trace loses required evidence')
    args = p.parse_args()
    if not 1 <= args.depth <= 256 or args.reloads < 3 or args.calls < 4 or args.states < 2 or args.max_growth_mib < 0:
        p.error('depth 1..256, reloads >=3, calls >=4, states >=2 and nonnegative growth bound required')
    raise SystemExit(0 if run(args) else 1)


if __name__ == '__main__':
    main()
