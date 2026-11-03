# PyPI release (not yet published)

Version: 0.2.1. Public distribution is authorized for preparation; credentials will
be configured separately. Do not publish automatically from commits or tags.
The GitHub repository remains private. PyPI wheels expose the included source.

## Build without changing the NPU environment

Use a separate release environment with `build` and `twine` installed. Do not
upgrade the SDK environment to install release tools.

```bash
python -m build
python -m twine check dist/fxbex-0.2.1-py3-none-any.whl dist/fxbex-0.2.1.tar.gz
```

Minimal wheel-only build (already verified on the server):

```bash
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/fxbex-0.2.1-py3-none-any.whl
python -c 'from fxbex import LLM; print(LLM)'
fxbex --help
```

Inspect archive members before publication: include FXBEX source and metadata,
not weights, SDK binaries, model packages, credentials, raw logs or local reports.
Choose and add the project's distribution license before public release; no
open-source license has been assumed on the owner's behalf. Core does not bundle
the private model adapter: users must separately install a compatible adapter,
SDK and weights. An import test is not an NPU inference test.

## Authenticate and publish later

Create/authorize a PyPI account, configure its token through the secure credential
store or Twine's interactive password prompt (username `__token__`). Never commit
credentials or put tokens in shell history. Confirm the name `fxbex` remains
available; an earlier 404 is not a reservation. Optionally test on TestPyPI first
using its separate credentials.

```bash
python -m twine upload dist/fxbex-0.2.1-py3-none-any.whl dist/fxbex-0.2.1.tar.gz
```

After successful publication, verify from a clean environment:

```bash
python -m pip install --index-url https://pypi.org/simple fxbex==0.2.1
python -c 'from fxbex import LLM; print(LLM)'
fxbex --help
```

Only then advertise `pip install fxbex` as available. Do not overwrite a published
version; increment the version for corrections. Actual Qwen NPU validation of the
new Python wrapper is deferred at the user's request.

Reference: https://packaging.python.org/en/latest/tutorials/packaging-projects/
