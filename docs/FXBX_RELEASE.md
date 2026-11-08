# fxbX distribution rename (0.7.0)

Public names: repository `smallstep-zzz/fxbX`, PyPI distribution `fxbx`,
CLI `fxbx`, Python `from fxbx import LLM, hug, Model, YOLO, ImageGeneration`.
The internal `fxbex` engine is intentionally shipped in the same wheel: saved
model adapters and artifact/plugin contracts refer to it. This is not a second
copy of the engine. Existing `.fxbex` files, `FXB_EX` manifests, settings paths
and plugin group `fxbex.adapters` remain unchanged. No NPU or cache migration.

Build/test in an isolated environment; then publish using the manual
`publish-pypi.yml` workflow with `confirm_version=0.7.0`. Git pushes do not publish.
For a new PyPI project configure a pending Trusted Publisher for **fxbx**:
owner `smallstep-zzz`, repository `fxbX`, workflow `publish-pypi.yml`, environment
blank (the current job has no GitHub environment). Availability of the name must
still be accepted by PyPI; a404 lookup is not a registration guarantee.
The previous `fxbex` project's publisher binding does not create this new project.

Until publication succeeds, source install:

```bash
python -m pip install 'git+https://github.com/smallstep-zzz/fxbX.git@main'
fxbx --version
```

After publication:

```bash
python -m pip install --upgrade fxbx
fxbx doctor
fxbx chat MODEL
```

Do not claim `pip install fxbx` works before checking the new PyPI release.
The transition does not require publishing/renaming existing model artifacts.

## Published release

Version0.7.0 was published successfully through GitHub Actions run42491523314.
A clean Python3.12 environment installed it from PyPI and verified `fxbx --version`
and public LLM/Model/YOLO/ImageGeneration/hug imports. Local CPU tests:145,
with1 skip. No new full-model NPU acceptance is claimed by this rename.
Before migrating an environment with the old distribution, use
`python -m pip uninstall fxbex`, then `python -m pip install fxbx==0.7.0`;
do not install/uninstall both distributions over shared engine files.
