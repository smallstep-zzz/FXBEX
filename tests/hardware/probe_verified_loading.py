"""Fresh producer qualification, strict consumer reload and stale-weight rejection.

Small frozen dense/stateful/prepared-value fixtures only, not pretrained-model
acceptance. Uses installed matching TCLMachine and fxbX, never downloads assets.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=('dense', 'stateful', 'prepared'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.model is None:
        # A direct GraphFXB load initializes an SDK device context owned by its
        # process. Exit that model worker before a different replay process loads.
        from fxbex.settings import npu_lock, lock_path
        result = {'status': 'running', 'models': [], 'worker_policy': 'one process per model; parent never imports SDK'}
        started = time.monotonic()
        try:
            with npu_lock() as lease:
                env = dict(os.environ, FXBEX_NPU_LOCK=str(lock_path()), FXBEX_NPU_LOCK_FD=str(lease))
                for name in ('dense', 'stateful', 'prepared'):
                    out = a.out / name
                    command = [sys.executable, str(Path(__file__).resolve()), '--out', str(out), '--model', name]
                    with (a.out / (name + '.log')).open('wb') as log:
                        child = subprocess.run(command, env=env, pass_fds=(lease,), stdout=log,
                                               stderr=subprocess.STDOUT, timeout=300)
                    row = json.loads((out / 'result.json').read_text())
                    if child.returncode or row['status'] != 'passed':
                        raise RuntimeError(f'{name} worker failed: {row.get("error", row)}')
                    result['models'].extend(row['models'])
                    (a.out / 'progress.json').write_text(json.dumps(result, indent=2))
            assert 'torch' not in sys.modules
            result['status'] = 'passed'
        except BaseException as error:
            result.update(status='failed', error_type=type(error).__name__, error=str(error))
            (a.out / 'traceback.txt').write_text(traceback.format_exc())
        result['seconds'] = time.monotonic() - started
        (a.out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'passed' else 1)
    import torch
    import furiosa.torch as sdk
    from fxbx import GraphFXB, SafeTensorWeights, build_model
    from fxbex.qualification import QualificationError
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(2); torch.manual_seed(611)

    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(8, 8)
        def forward(self, x):
            return self.linear(x)

    class Cell(Dense):
        def forward(self, x, state):
            return self.linear(x) + state

    class Prepared(torch.nn.Module):
        def forward(self):
            return torch.arange(8, dtype=torch.int64)

    result = {'status': 'running', 'scope': 'fixed-shape frozen synthetic fixtures; not full pretrained-model support', 'models': []}
    started = time.monotonic()
    try:
        for name, model in (('dense', Dense()), ('stateful', Cell()), ('prepared', Prepared())):
            if name != a.model:
                continue
            model = model.bfloat16().eval().requires_grad_(False)
            root = a.out
            output = root / 'model.fxb'
            x = torch.randn(4, 8, dtype=torch.bfloat16)
            stateful, prepared = name == 'stateful', name == 'prepared'
            args = () if prepared else (x, torch.zeros_like(x)) if stateful else (x,)
            options = {'state_inputs': {1: 'memory'}, 'state_updates': {'memory': 0}, 'user_output_indices': [0]} if stateful else {}
            tick = time.monotonic()
            report = build_model(model, output, example_inputs=args, concurrency=2,
                                 verification={'validation_inputs': [] if prepared else [(x + .5,)],
                                               'atol': .02, 'rtol': .02, 'repeats': 3, 'replay_timeout': 180}, **options)
            assert report['status'] == 'verified', report
            producer = report['result']; receipt = Path(producer['qualification'])
            checkpoint = Path(str(output) + '.verified') / 'weights'
            weights = SafeTensorWeights(checkpoint) if not prepared else {}
            row = {'name': name, 'producer_status': producer['status'], 'build_and_qualification_s': time.monotonic() - tick,
                   'producer_calls': len(producer['replay']['calls']),
                   'producer_cpu_routes': sum(c['cpu_routes'] for c in producer['replay']['calls']),
                   'npu_task_steps': sum(len(c['npu_steps']) for c in producer['replay']['calls']),
                   'static_preparation_steps_without_task': sum(len(c['constant_steps_without_task']) for c in producer['replay']['calls']),
                   'strict_reloads': 0, 'calls': 0, 'receipt': str(receipt)}
            for reload in range(3):
                with GraphFXB(output, weights=weights, qualification=receipt) as runtime:
                    assert runtime.qualification['status'] == 'verified'
                    expected_state = torch.zeros_like(x)
                    for i in range(5):
                        v = x + i * .125
                        with torch.no_grad():
                            expected = model() if prepared else model(v, expected_state) if stateful else model(v)
                        actual, = runtime.predict(() if prepared else (v,))
                        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                        if stateful:
                            expected_state = expected
                        row['calls'] += 1
                    try:
                        runtime.predict(() if prepared else (x,), output_device='furiosa')
                    except ValueError:
                        row['unqualified_output_device_rejected'] = True
                    else:
                        raise AssertionError('CPU-only receipt allowed unchecked NPU output exposure')
                    assert not runtime.failed
                assert runtime.completion_known
                row['strict_reloads'] += 1
            stale = root / 'stale-receipt.json'
            document = json.loads(receipt.read_text())
            document['bindings']['consumer']['source_sha256'] = '0' * 64
            stale.write_text(json.dumps(document))
            with patch.object(sdk, 'Runnable', side_effect=AssertionError('must reject before device upload')) as construct:
                try:
                    with GraphFXB(output, weights=weights, qualification=stale):
                        pass
                except QualificationError:
                    row['stale_consumer_rejected_before_allocation'] = construct.call_count == 0
                else:
                    raise AssertionError('stale consumer receipt accepted')
            if not prepared:
                bad = root / 'changed-weights'; shutil.copytree(checkpoint, bad)
                file = next(bad.glob('[0-9]*.safetensors'))
                values = load_file(str(file)); key = next(iter(values)); values[key] = values[key] + 1
                save_file(values, str(file))
                with patch.object(sdk, 'Runnable', side_effect=AssertionError('must reject before device upload')) as construct:
                    try:
                        with GraphFXB(output, weights=SafeTensorWeights(bad), qualification=receipt):
                            pass
                    except QualificationError:
                        row['same_shape_changed_weights_rejected_before_allocation'] = construct.call_count == 0
                    else:
                        raise AssertionError('changed weight values inherited old qualification')
            row['status'] = 'passed'; result['models'].append(row)
            (a.out / 'progress.json').write_text(json.dumps(result, indent=2))
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
