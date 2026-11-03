# FXBEX

Linux CLI for packaging and running precompiled Furiosa RNGD artifacts.
`.fxbex` is an independent format, **not an official Furiosa FXB**.

## Install

Python3.10+ is required. Install a compatible Furiosa SDK separately for NPU execution.

```bash
python -m pip install fxbex
fxbex --help
```

For local development from a checkout, use `python -m pip install -e .` instead.

To avoid changing any existing SDK dependencies:

```bash
PYTHON=python3.10 bash scripts/install.sh
```

This package does not install/upgrade Furiosa SDK, PyTorch or drivers. Build/inspect
and integrity-only validation use the Python standard library; NPU execution requires
the package's exact compatible SDK. Version 0.3.0 is published on
[PyPI](https://pypi.org/project/fxbex/0.3.0/); pin with `python -m pip install fxbex==0.3.0` for reproducibility.

## Commands

```bash
fxbex build ./compiled_model -o model.fxbex
fxbex build ./compiled_model -o model-standalone.fxbex --standalone
fxbex inspect model.fxbex
fxbex validate model.fxbex
fxbex run model.fxbex
fxbex chat model.fxbex
fxbex serve model.fxbex --port 8000
fxbex doctor
```

`run` starts chat for a chat adapter. For a tensor model, supply
`--input inputs.json -o result.json`. `chat` supports `/reset` and `/quit`.
`serve` binds localhost and supports non-streaming `/v1/chat/completions` for chat
adapters and `/predict` for tensor adapters. `fxbex ui --port 7861` is optional and
requires Gradio. Every command has `--help`.

`validate` checks checksums, external weights, SDK versions, installed adapter
contracts and RNGD detection; it **does not execute inference**. To inspect on a
machine without SDK/NPU use `fxbex validate model.fxbex --integrity-only`.
`--weights-dir /path/to/snapshot` relocates external weights, with hash validation.

The builder packages existing compiled artifacts; it does not compile an arbitrary
HF model. Kernel load fails on missing or mismatched artifacts, never silently
recompiles. Use only trusted packages and installed plugins.

## Model Forge integration

FXBEX core has no `compiler_engine` dependency. Generic `runnable_dag` packages
run directly. Model-specific code belongs to separately installed adapter plugins:

```bash
cd /root/rngd-model-forge
bash tests/install_fxbex.sh /root/FXBEX
```

This installs the `qwen35_replay` plugin via the `fxbex.adapters` Python entry point
and writes a shared NPU lock path in `~/.config/fxbex/config.json`, coordinating
with the existing Model Forge web workers. Model Forge's **FXBEX package** tab
calls this installed runtime in a subprocess; unloading releases the process.

A plugin class receives `(extracted_root, report_dir, ExitStack, weights_dir=None)`
and provides `chat` and/or `predict`; `preflight(root, weights_dir=None)` performs
static checks. Packages name an installed entry point, never an arbitrary import
or embedded Python program. Conflicting or missing entry points fail explicitly.

Default reports: `~/.cache/fxbex/runs`. Configure `reports_dir` and `npu_lock` in
`~/.config/fxbex/config.json`; `FXBEX_CONFIG` and `FXBEX_NPU_LOCK` can override paths.
The integration installer deliberately shares Model Forge's existing lock.

## Scope

Current built-in runtime supports fixed tensor DAGs on RNGD fusion8. Qwen's plugin
is text-only BF16 with512-token capacity. Previous verified Qwen package reload
used zero compiler calls; first load still takes about3minutes for weight preparation
and graph reconstruction. Other model adapters and automatic memory planning are
not implemented. Historical validation records in `docs/` predate this repository
split; current split validation is recorded separately.

Tests: `python3.10 -m unittest discover -s tests`.

## Reusable Gradio model page

Copy `examples/model_page.py`, set `PACKAGE = Path('fxbex/model.fxbex')`, then
run it on the SDK host. Keep delivery packages in your application's `fxbex/`
folder: `fxbex build ./compiled_model -o fxbex/model.fxbex`.
The page lists saved packages, copies uploaded packages there without replacing
existing files, provides Load/Unload, measures request wall time, and displays a
Python load/run/release example with the selected filename below the controls.
Use `mode='chat'` or `mode='predict'`; an agent can supply `controls(run)` for
image/upscale-specific inputs and outputs without rewriting the lifecycle shell.
That callback calls `run('predict', named_inputs)` and receives `(result, timing)`.
Input conversion and output rendering must match a validated model adapter;
the template does not add model support. Existing Model Forge embeds this same
shell with its coordinated service. Packages remain excluded from Git.

## Python LLM API

```python
from fxbex import LLM

package = '/root/rngd-model-forge/fxbex/RedHatAI--Qwen3.5-9B-FP8-dynamic__text-bf16__fusion8__ctx512__sdk2026.4.0.fxbex'
with LLM('RedHatAI/Qwen3.5-9B-FP8-dynamic', fxbex=package) as llm:
    print(llm.generate('What is 2+2? Reply briefly.')['text'])
    print(llm.converse('What is 2+2?')['text'])
    print(llm.converse('Multiply that answer by 3.')['text'])
    llm.reset()
```

`fxb=package` is an alias for `fxbex=package`; supply exactly one. `LLM(fxbex=package)`
reads model identity from the package. Explicit model names must match; naming
Qwen3-8B does not turn a Qwen3.5-9B package into that model. `chat(messages)` accepts
explicit message history. Results are dictionaries containing `text` and available
runtime metrics. Without a `with` block, call `llm.close()` (or `unload()`). Model
execution runs in a persistent subprocess; closing releases its NPU allocations.

This API accepts FXBEX content even if the filename ends in `.fxb`, but **official
Furiosa FXB content is not supported by this loader**; use `furiosa_llm.LLM` for it.
The compatible SDK, original weights and model adapter plugin are still required.
No model download, automatic compilation, or CPU fallback is introduced.

### Wheel installation

Build with `python -m pip wheel --no-deps --no-build-isolation . -w dist`, then:

```bash
python -m pip install dist/fxbex-0.3.0-py3-none-any.whl
```

`pip install fxbex` installs the published Python core.
GitHub remains private. Publishing a wheel to PyPI exposes its source publicly;
Maintainer authentication is needed only to publish future releases, not to install.

Release preparation and keyless GitHub OIDC authentication: [docs/PYPI_RELEASE.md](docs/PYPI_RELEASE.md).
The Python core is Apache-2.0 licensed; Furiosa SDK and model assets are not bundled.
Additional NPU testing of the LLM wrapper and isolated portable reload was skipped at the user's request. API unit tests and portable integrity/source/adapter preflight pass; this does not prove isolated portable inference.

## Single-file model delivery (0.3.0)

A small reference package is not a complete model: it can reference external
weights and an installed adapter. `build --standalone` includes indexed weights,
tokenizer/config files, but an external adapter may still be required.
A portable delivery additionally embeds the exact Python model runtime under
`python/`, selected by `runtime.bundled_python`. Use only trusted packages: loading
this form executes its Python code. Checksums detect corruption, not authorship.
The built-in loader uses an isolated worker and rejects already-imported runtime
module conflicts. Package Python must not replace the `fxbex` core.

For the validated Model Forge Qwen source:

```bash
cd /root/rngd-model-forge
python -m tools.export_portable_qwen fxbex/model.fxbex -o /large-disk/model-portable.fxbex
```

This checks pinned project sources and embeds checkpoint files, tokenizer,
compiled kernels and Python adapter. It avoids downloads and separate Model Forge
installation on the receiving host. A compatible RNGD driver, Furiosa SDK and
Python dependencies are still host prerequisites; they are not bundled or silently
installed. Thus `pip install fxbex` plus one file is sufficient **only on an already
compatible RNGD host**, with FXBEX 0.3.0 or newer. PyPI 0.2.1
does not support the bundled adapter feature. Full relocation NPU acceptance is
required before advertising a model as portable; static checks are not inference.
Allow space for both the package and its extracted contents on the temporary disk.
