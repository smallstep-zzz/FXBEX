"""Check cached official FXB pass-through and generation without model downloads.

Official LLM acceptance is separate from static GraphFXB qualification. This probe
does not provide per-step NPU trace evidence for the official native runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fxb', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--device', default='npu:3')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not a.fxb.is_file() or not a.snapshot.is_dir():
        p.error('existing local FXB and snapshot required')
    a.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = {'status': 'running', 'route': 'official_llm', 'fxb': str(a.fxb),
              'snapshot': str(a.snapshot), 'scope': 'official cached load and deterministic functional generation',
              'npu_step_evidence': 'not_collected', 'builder_attempts': [], 'outputs': []}
    started = time.monotonic(); model = None
    try:
        from fxbx import hug
        from fxbex.settings import npu_lock
        import furiosa.llm_native.build as official_build
        from unittest.mock import patch
        def forbidden(*args, **kwargs):
            result['builder_attempts'].append('build_fxb')
            raise AssertionError('Cached FXB loading must not invoke the official builder')
        # Native default routing is tested without fxbex=False overriding it.
        with npu_lock(), patch.object(official_build, 'build_fxb', side_effect=forbidden):
            from furiosa_llm import LLM, SamplingParams
            tick = time.monotonic()
            model = hug(LLM)(str(a.snapshot), fxb=str(a.fxb), devices=[a.device], max_model_len=512)
            result['load_seconds'] = time.monotonic() - tick
            history = []
            for prompt, expected in [('Reply with only the number: what is 2+2?', '4'),
                                     ('Reply with exactly this word: READY', 'READY'),
                                     ('Reply with exactly this word: READY', 'READY')]:
                history.append({'role': 'user', 'content': prompt})
                tick = time.monotonic()
                response = model.chat(history, sampling_params=SamplingParams(temperature=0, max_tokens=32),
                                      chat_template_kwargs={'enable_thinking': False})
                text = response[0].outputs[0].text
                result['outputs'].append({'prompt': prompt, 'expected': expected, 'text': text,
                                          'request_seconds': time.monotonic() - tick,
                                          'reported_cached_tokens': getattr(response[0], 'num_cached_tokens', None)})
                if text.strip() != expected:
                    raise AssertionError('Official model output did not match the declared deterministic check')
                history.append({'role': 'assistant', 'content': text})
            model.shutdown(); model = None
        result.update(status='passed', generation_host_median_seconds=statistics.median(r['request_seconds'] for r in result['outputs']))
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (a.out / 'traceback.txt').write_text(traceback.format_exc())
    finally:
        if model is not None:
            try:model.shutdown()
            except BaseException as cleanup:
                result.update(status='failed', cleanup_error=str(cleanup))
    result['seconds'] = time.monotonic() - started
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
