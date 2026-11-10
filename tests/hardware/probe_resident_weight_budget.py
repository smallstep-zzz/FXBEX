"""Synthetic 8GiB residency using distinct targets over one small checkpoint.

Compiles one qualified dense EDF, reuses its exact ABI in a long fixed-shape
chain and maps each distinct weight target to the same immutable source tensor.
This checks residency/admission/cleanup, not 9B checkpoint diversity or disk I/O.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile


def memory(label):
    rss = next(int(x.split()[1]) * 1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
    data = json.loads(subprocess.check_output(['furiosa-smi', 'status', '--format', 'json'], text=True))
    return {'label': label, 'rss_bytes': rss,
            'npu_bytes': {d['device']: d['memory']['DRAM']['used_size'] for d in data}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--layers', type=int, default=256)
    parser.add_argument('--reloads', type=int, default=2)
    args = parser.parse_args()
    if args.width < 1 or args.layers < 1 or args.reloads < 1:
        parser.error('positive dimensions and reload count required')
    resident_bytes = args.layers * args.width**2 * 2 + args.width * 2
    if resident_bytes > 12 * 1024**3:
        parser.error('synthetic persistent weights must stay below12GiB')
    root = args.out; root.mkdir(parents=True, exist_ok=False)
    os.environ['FXBX_TRACE_STEPS'] = '1'
    import torch
    import tclmachine
    from fxbx import GraphFXB, SafeTensorWeights
    from fxbex.execution_fxb import read_execution_plan, export_execution_plan
    from safetensors.torch import save_file
    from furiosa.torch.profiler import Profiler, RecordFunction
    torch.set_num_threads(2)
    result = {'status': 'running', 'scope': 'synthetic replicated resident targets, not a distinct9B checkpoint or I/O benchmark',
              'stage': 'compile', 'width': args.width, 'layers': args.layers, 'resident_weight_bytes': resident_bytes,
              'budget_bytes': 16 * 1024**3, 'memory': []}
    started = time.monotonic()
    try:
        model = torch.nn.Linear(args.width, args.width).bfloat16().eval().requires_grad_(False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(args.width, dtype=torch.bfloat16) * .99); model.bias.zero_()
        x = torch.linspace(-1, 1, args.width, dtype=torch.bfloat16).repeat(4, 1).contiguous()
        original = root / 'one.fxb'
        tclmachine.export_model(model, (x,), original, compile_workers=1, artifact_root=root / 'edfs')
        plan = read_execution_plan(original)
        entry = plan['entries']['forward']
        if len(entry['steps']) != 1 or len(entry['outputs']) != 1:
            raise ValueError('expected one dense step')
        source_step = entry['steps'][0]; source_output = source_step['outputs'][0]
        weight_name = next(k for k, s in plan['tensors'].items() if s.get('target') == 'weight')
        bias_name = next(k for k, s in plan['tensors'].items() if s.get('target') == 'bias')
        weight_spec = plan['tensors'].pop(weight_name)
        bias_spec = plan['tensors'][bias_name]; bias_spec['target'] = 'shared_bias'
        names = {'shared_bias': 'bias'}; previous = entry['inputs'][0]; steps = []
        for i in range(args.layers):
            weight = f'weight:{i}'; target = f'W_{i:04d}'; output = f'layer:{i}'
            plan['tensors'][weight] = {**weight_spec, 'target': target}
            plan['tensors'][output] = copy.deepcopy(plan['tensors'][source_output])
            bindings = {weight_name: weight, entry['inputs'][0]: previous}
            steps.append({'kernel': source_step['kernel'], 'inputs': [bindings.get(k, k) for k in source_step['inputs']], 'outputs': [output]})
            previous = output; names[target] = 'weight'
        del plan['tensors'][source_output]
        entry.update(steps=steps, outputs=[previous])
        plan['memory_policy'] = {'budget_bytes': 16 * 1024**3, 'reserve_bytes': 1024**3, 'safety_margin_bytes': 0}
        plan['source_metadata'] = {'architecture': 'SyntheticReplicatedResidentLinear', 'layers': args.layers,
                                   'note': 'each GPU target is distinct; same immutable tensor backs every checkpoint mapping'}
        extraction = root / 'extract'; extraction.mkdir()
        with zipfile.ZipFile(original) as archive:
            for i, kernel in enumerate(plan['kernels'].values()):
                path = extraction / f'{i}.edf'; path.write_bytes(archive.read(kernel.pop('member'))); kernel['edf_path'] = str(path)
        artifact = root / 'resident.fxb'; export_execution_plan(plan, artifact)
        checkpoint = root / 'checkpoint'; checkpoint.mkdir()
        save_file(model.state_dict(), str(checkpoint / 'weights.safetensors'))
        result['checkpoint_bytes'] = (checkpoint / 'weights.safetensors').stat().st_size
        result['fxb_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        result['stage'] = 'cpu_reference'; fixtures = []
        with torch.no_grad():
            for inputs in (x, x + .25):
                expected = inputs
                for _ in range(args.layers):
                    expected = torch.nn.functional.linear(expected, model.weight, model.bias)
                fixtures.append((inputs, expected))
        result['memory'].append(memory('before_load'))
        calls = []; result['stage'] = 'replay'
        trace = root / 'session-trace.json'
        with Profiler(str(trace)):
            for reload in range(args.reloads):
                start = time.monotonic()
                with GraphFXB(artifact, weights=SafeTensorWeights(checkpoint, names=names)) as runtime:
                    loaded = memory(f'loaded:{reload}'); result['memory'].append(loaded)
                    if max(loaded['npu_bytes'].values()) < resident_bytes:
                        raise AssertionError('SMI did not observe the requested distinct resident weights')
                    result.setdefault('load_seconds', []).append(time.monotonic() - start)
                    for inputs, expected in fixtures:
                        label = f'audit_call::{len(calls)}'; tick = time.monotonic()
                        with RecordFunction(label, propagate=True): actual, = runtime.predict((inputs,))
                        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                        calls.append({'label': label, 'reload': reload, 'host_seconds': time.monotonic() - tick})
                result['memory'].append(memory(f'closed:{reload}'))
        events = json.loads(trace.read_text())['traceEvents']; cpu = sum('run_by_cpu' in e.get('name', '') for e in events)
        if cpu: raise AssertionError('CPU model routes observed')
        expected = {f"execution_plan::forward::{i}::{s['kernel']}" for i, s in enumerate(steps)}
        for call in calls:
            frames = [e for e in events if e.get('name') == call['label'] and e.get('ph') == 'X']
            if len(frames) != 1: raise AssertionError('call frame missing')
            low = frames[0]['ts']; high = low + frames[0]['dur']
            window = [e for e in events if 'ts' in e and low <= e['ts'] and e['ts'] + e.get('dur', 0) <= high]
            spans = {e['name'] for e in window if e.get('name', '').startswith('execution_plan::')}
            tasks = {e['name'].split('::Task::', 1)[1] for e in window if '::Task::execution_plan::' in e.get('name', '')}
            if spans != expected or tasks != expected:raise AssertionError(f'evidence missing: expected{len(expected)}, spans{len(spans)}, tasks{len(tasks)}')
        result.update(status='passed', calls=calls, call_bounded_steps=len(calls) * len(steps), cpu_routes=cpu)
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (root / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__': main()
