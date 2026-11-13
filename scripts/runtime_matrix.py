#!/usr/bin/env python3
"""Run a trusted FXBX sample in each isolated SDK environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


PACKAGES = (
    'fxbx', 'torch', 'furiosa-torch', 'furiosa-tcc', 'furiosa-kernels',
    'transformers', 'onnx', 'onnx2torch', 'torchvision',
)
VERSION_PROBE = """
import importlib.metadata as metadata
import json
import platform
import sys
installed = {
    dist.metadata['Name'].lower().replace('_', '-'): dist.version
    for dist in metadata.distributions()
    if dist.metadata['Name']
}
print(json.dumps({'python': platform.python_version(),
                  **{name: installed.get(name) for name in sys.argv[1:]}}))
"""


def run_case(python, package, directory, timeout, env):
    directory.mkdir()
    report = directory / 'reports'
    result = {'environment': python.parent.parent.name, 'python_executable': str(python),
              'status': 'environment_error', 'npu_trace': 'unknown', 'directory': str(directory)}
    started = time.monotonic()
    env = dict(env, FXBEX_WORKER_PYTHON=str(python))
    try:
        probe = subprocess.run([str(python), '-I', '-c', VERSION_PROBE, *PACKAGES],
                               env=env, capture_output=True, text=True, timeout=30, check=True)
        result['versions'] = json.loads(probe.stdout)
        with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
            completed = subprocess.run(
                [str(python), '-I', '-m', 'fxbex.cli', 'run', str(package),
                 '--sample', '--report-dir', str(report)],
                cwd=directory, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                stderr=stderr, timeout=timeout,
            )
        result['returncode'] = completed.returncode
        if completed.returncode:
            result['status'] = 'failed'
            with (directory / 'stderr.log').open('rb') as log:
                log.seek(max(0, log.seek(0, 2) - 8192))
                lines = log.read().decode(errors='replace').strip().splitlines()
            result['error'] = lines[-1][:600] if lines else 'See runtime.log'
        else:
            receipt = report / 'result.json'
            if not receipt.is_file():
                result.update(status='missing_result', error='CLI returned without result.json')
            else:
                data = json.loads(receipt.read_text())
                result.update(status='passed', result_file=str(receipt))
                if data.get('current_request_traced') is True and (data.get('npu_tasks') or 0) > 0:
                    result['npu_trace'] = 'confirmed'
                    result['npu_tasks'] = data['npu_tasks']
                if type(data.get('inference_s')) in (int, float):
                    result['inference_seconds'] = data['inference_s']
    except subprocess.TimeoutExpired:
        result.update(status='timeout', error='Execution or interpreter probe timed out')
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        result['error'] = str(exc)[:600]
    result['seconds'] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path, help='Local .fxbx or .fxbex with an embedded sample')
    parser.add_argument('--env-dir', type=Path, default=Path(__file__).resolve().parent / 'venvs')
    parser.add_argument('--env', action='append', help='Run only this environment name. Repeat to select several.')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds allowed per environment (default 1800)')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'runs')
    args = parser.parse_args()
    package = args.package.resolve()
    if package.suffix not in ('.fxbx', '.fxbex') or not package.is_file():
        parser.error('Provide an existing .fxbx or .fxbex file')
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    names = sorted(set(args.env or ()))
    if not args.env:
        names = sorted(p.name for p in args.env_dir.glob('*') if (p / 'bin/python').is_file())
    if not names or any(name in ('.', '..') or Path(name).name != name for name in names):
        parser.error('Provide valid environment names or populate --env-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='matrix-', dir=args.output_dir.resolve()))
    print(f'Reports {output}', flush=True)
    env = os.environ.copy()
    # Force every child to run in its selected interpreter and share the normal fxbX NPU lock.
    for key in ('FXBEX_WORKER_PYTHON', 'FXBEX_NPU_LOCK_FD', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env['FURIOSA_REQUIRE_NPU'] = '1'
    summary = {'package': str(package), 'host': os.uname().nodename, 'results': []}
    for name in names:
        python = args.env_dir.resolve() / name / 'bin/python'
        row = run_case(python, package, output / name, args.timeout, env)
        summary['results'].append(row)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"{name:18} {row['status']:16} {row['seconds']:7.2f}s  NPU trace {row['npu_trace']}"
              + (f"  {row['error']}" if 'error' in row else ''), flush=True)
    return int(any(row['status'] != 'passed' for row in summary['results']))


if __name__ == '__main__':
    sys.exit(main())
