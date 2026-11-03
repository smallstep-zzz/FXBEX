# FXBEX

Linux CLI for packaging and running precompiled Furiosa RNGD artifacts.
`.fxbex` is an independent format, **not an official Furiosa FXB**.

## Install

Python3.10+ is required. Install a compatible Furiosa SDK separately for NPU execution.

```bash
cd /root/FXBEX
pip install -e .
fxbex --help
```

To avoid changing any existing SDK dependencies:

```bash
PYTHON=python3.10 bash scripts/install.sh
```

This repository does not install/upgrade Furiosa SDK, PyTorch or drivers. Build/inspect
and integrity-only validation use the Python standard library; NPU execution requires
the package's exact compatible SDK. No PyPI publication is claimed.

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
