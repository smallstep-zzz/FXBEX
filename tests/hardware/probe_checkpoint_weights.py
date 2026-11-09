"""Real native FXB replay with header-first weights, state and repeated calls.

Uses the existing small shared-linear FXB fixture; never compiles or imports the
producer. Its checkpoint fixture uses seed 122, as probe_borrowed_state_program.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from fxbx import GraphFXB, SafeTensorWeights
from furiosa.torch.profiler import Profiler
import furiosa.torch as sdk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(122)
    core = torch.nn.Linear(8, 8).bfloat16().eval().requires_grad_(False)
    for name, value in core.state_dict().items():
        save_file({'core.' + name: value}, str(a.out / (name + '.safetensors')))
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    token = torch.randn(1, 8, dtype=torch.bfloat16)
    original = {k: v.clone() for k, v in core.state_dict().items()}
    with torch.no_grad():
        expected = core(x)
    bad = a.out / 'bad'; bad.mkdir()
    save_file({'core.weight': torch.zeros(9, 8, dtype=torch.bfloat16),
               'core.bias': core.bias}, str(bad / 'weights.safetensors'))
    invalid = SafeTensorWeights(bad)
    with patch.object(sdk, 'Runnable', side_effect=AssertionError('must not allocate')) as construct:
        try:
            with GraphFXB(a.fxb, weights=invalid): pass
        except ValueError as error:
            assert 'checkpoint shape/dtype' in str(error)
        else: raise AssertionError('metadata mismatch accepted')
        assert construct.call_count == 0 and invalid.loaded_tensors == 0
    trace = a.out / 'trace.json'; rows = []; elapsed = []
    import os
    os.environ['FXBX_TRACE_STEPS'] = '1'
    for reload in range(3):
        start = time.perf_counter(); weights = SafeTensorWeights(a.out)
        metadata_s = time.perf_counter() - start
        assert weights.loaded_tensors == 0
        start = time.perf_counter()
        with GraphFXB(a.fxb, weights=weights) as model:
            load_s = time.perf_counter() - start
            assert model.weights is weights and weights.loaded_tensors == 2
            # Only the final replay is profiled; warm timing excludes profiling.
            from contextlib import nullcontext
            with Profiler(str(trace)) if reload == 2 else nullcontext():
                actual, = model.run('prefill', (x,))
                torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                state = expected
                for i in range(5):
                    start = time.perf_counter()
                    actual, = model.run('decode', (token,))
                    elapsed.append(time.perf_counter() - start) if reload < 2 else None
                    state = core(token).expand(4, 8).contiguous() + state
                    torch.testing.assert_close(actual, state, atol=.02, rtol=.02)
                model.reset()
                actual, = model.run('prefill', (x,))
                torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
            assert weights.loaded_tensors == 2
        rows.append({'metadata_s': metadata_s, 'load_s': load_s, 'tensor_value_reads': weights.loaded_tensors})
    assert all(torch.equal(core.state_dict()[k], v) for k, v in original.items())
    assert not any(n == 'tplmachine' or n.startswith('tplmachine.') for n in sys.modules)
    events = json.loads(trace.read_text())['traceEvents']
    names = {e['name'] for e in events if e.get('name', '').startswith('execution_plan::')}
    observed = {e['name'].split('::Task::', 1)[1] for e in events if '::Task::execution_plan::' in e.get('name', '')}
    assert names and names == observed
    assert not any(e.get('name') == 'run_by_cpu' for e in events)
    result = {'status': 'passed', 'scope': 'small fixed-shape native FXB; not full-model or throughput proof',
              'replays': rows, 'calls': 21, 'npu_confirmed_steps': len(names), 'cpu_routes': 0,
              'bad_metadata_rejected_before_device_load': True, 'producer_imported': False,
              'whole_checkpoint_materialized_on_cpu': False,
              'weight_policy': 'strict dtype/shape; synchronous one-tensor upload; no quantization conversion',
              'warm_completed_host_call_median_s': statistics.median(elapsed),
              'report_dir': str(a.out)}
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
