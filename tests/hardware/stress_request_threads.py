"""Four host callers over independent states; one session serializes NPU work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--calls', type=int, default=100)
    a = p.parse_args()
    if a.threads < 2 or a.calls < 1:
        p.error('threads >=2 and calls >=1 required')
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    from fxbx import GraphFXB, SafeTensorWeights
    torch.set_num_threads(2)
    weights = SafeTensorWeights(a.weights)
    weight, bias = weights['core.weight'], weights['core.bias']
    result = {'status': 'running', 'scope': 'four host callers; session RLock serializes NPU work; fixed synthetic fixture',
              'threads': a.threads, 'calls_per_thread': a.calls}
    started = time.monotonic()
    try:
        with GraphFXB(a.fxb, weights=SafeTensorWeights(a.weights), max_request_states=a.threads) as runtime:
            states = [runtime.new_state() for _ in range(a.threads)]
            def client(index):
                reference = torch.zeros(4, 8, dtype=torch.bfloat16)
                queues = []
                for i in range(a.calls):
                    if i % 20 == 0:
                        runtime.reset(state=states[index]); reference.zero_()
                    entry = 'prefill' if i % 10 == 0 else 'decode'
                    shape = (4, 8) if entry == 'prefill' else (1, 8)
                    x = torch.full(shape, .25 * (index - 2) + .125 * (i % 3), dtype=torch.bfloat16)
                    with torch.no_grad():
                        reference = torch.nn.functional.linear(x, weight, bias).expand(4, 8).contiguous() + reference
                    actual, = runtime.run(entry, (x,), state=states[index])
                    torch.testing.assert_close(actual, reference, atol=.02, rtol=.02)
                    queues.append(states[index].last_timing['queue_wait_s'])
                return {'thread': index, 'calls': a.calls, 'queue_wait_median_s': statistics.median(queues)}
            with ThreadPoolExecutor(max_workers=a.threads) as pool:
                result['clients'] = list(pool.map(client, range(a.threads)))
            for state in states:
                state.close()
        result.update(status='passed', cpu_reference_comparisons=a.threads * a.calls,
                      expected_calls=a.threads * a.calls, isolation_and_reset='passed')
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__': main()
