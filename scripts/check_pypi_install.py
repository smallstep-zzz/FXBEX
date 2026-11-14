"""Verify what an unpinned pip upgrade actually installs from the public index."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import venv


def main(expected):
    env = {name: value for name, value in os.environ.items() if not name.startswith('PIP_')}
    env['PIP_CONFIG_FILE'] = os.devnull
    with tempfile.TemporaryDirectory() as directory:
        venv.EnvBuilder(with_pip=True).create(directory)
        python = Path(directory) / 'bin/python'
        deadline = time.monotonic() + 540
        while (remaining := deadline - time.monotonic()) > 0:
            result = subprocess.run(
                [python, '-m', 'pip', 'install', 'fxbx', '--upgrade', '--no-cache-dir'],
                cwd=directory, env=env, timeout=remaining,
            )
            if result.returncode == 0:
                installed = subprocess.check_output(
                    [python, '-I', '-c', 'from importlib.metadata import version; print(version("fxbx"))'],
                    cwd=directory, env=env, text=True,
                ).strip()
                print(f'Public pip installed {installed}; expected {expected}', flush=True)
                if installed == expected:
                    subprocess.run([python, '-m', 'pip', 'check'], cwd=directory, env=env, check=True)
                    subprocess.run([python.parent / 'fxbx', '--help'],
                                   cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL)
                    return
            time.sleep(max(0, min(30, deadline - time.monotonic())))
    raise SystemExit(f'Public pip install did not verify fxbx {expected}')


if __name__ == '__main__':
    main(sys.argv[1])
