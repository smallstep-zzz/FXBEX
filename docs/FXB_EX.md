> Historical implementation record from the original monorepo. For current installation/CLI see ../README.md. Model-specific capture is now `python -m forge_fxbex capture-qwen` in the producer project.

# FXB_EX v0.1

`.fxbex` is a project package, not a Furiosa `.fxb` and not consumed by `furiosa-llm --fxb`. It reuses the installed SDK's typed `Runnable.serialize` / `deserialize` interface; artifacts are labelled `sdk-runnable`, never falsely labelled raw EDF. The installed SDK runtime and its compatibility contract remain required.

## Install and use

From this checkout on the RNGD server:

```bash
python3.10 -m pip install --no-deps --no-build-isolation -e .
fxbex doctor
fxbex inspect /path/model.fxbex
fxbex validate /path/model.fxbex
fxbex chat /path/qwen.fxbex
fxbex predict /path/model.fxbex --input inputs.json --output result.json
fxbex serve /path/qwen.fxbex --port 8000
fxbex ui --port 7861
```

`python3.10 -m fxbex` works without installing the console command. `pip install fxbex` from PyPI is not a published installation method. Existing Furiosa/torch dependencies are neither installed nor upgraded by these instructions. UI requires the existing Gradio installation. Serve/UI bind localhost; use an SSH tunnel remotely. Ctrl-C closes the CLI server; `/quit` exits chat, `/reset` clears conversation history. Unload other web models first; the canonical project NPU lock prevents simultaneous fusion8 loading.

`serve` implements non-streaming `/v1/chat/completions`, `/v1/models`, `/status`, and `/predict`. It is a serialized single-model local service, not a complete OpenAI API. Clients must supply conversation history. The terminal/UI preserve their own history. UI supports chat packages; generic tensor packages use `predict`. Image generation adapters and cross-process model-manager commands are not implemented in v0.1.

## Runtime adapters

- `runnable_dag` deserializes the SDK runnables stored in the package and executes their declared tensor plan.
- `qwen35_replay` reuses the recorded runnables for the exact request contracts. A request that does not match the saved contract fails.

Use `chat/run/serve --weights-dir /path/to/pinned/snapshot` to locate external weights when needed. The required hashes are verified.

## Acceptance gates

Package integrity, installed compatibility, kernel contract, actual NPU trace, numerical equivalence, conversation state and repeated latency are separate gates. A successful builder is not model acceptance. Runtime diagnostics and raw traces remain under `reports/rngd_porting/runs/`; raw traces are omitted from the package. Never silently rebuild on load, invoke a model-name registry substitute, or treat device placement as NPU proof.

The implementation uses public Python SDK entrypoints and SDK-shipped type declarations. No proprietary binary analysis or modifications are performed. The `_C` namespace used by the SDK's typed Runnable implementation is version-sensitive; packages pin the SDK and relevant installed source hashes, and reject mismatches. No universal cross-version ABI promise.

The reused Qwen diagnostic wrapper still labels graph preparation `NPU COMPILE START/READY`; during replay these labels enclose artifact lookup rather than native compilation. Use the replay compiler counters and compiler guard to determine whether compilation occurred. Frontend preparation is a remaining load-time cost, not a warm inference compile.

## Verified initial results (2026-11-03)

See `FXB_EX_VALIDATION.json` for measured values and report paths. A 2-kernel DAG package produced bitwise-equal CPU results after reload, with 4 NPU tasks, zero SDK CPU routes and zero compiler calls (warm device execution ~0.29–0.43 ms). The Qwen package contains 25 distinct serialized runnable contracts, about4.3MB excluding its external checkpoint. Fresh-process Qwen replay served 511 recorded compiler requests with zero new compiler calls. Reload acceptance: cosine0.9999294, RMSE0.0447029, top1 match, 1,562 NPU tasks, zero SDK CPU routes and zero unknown graphs. API conversation returned4 then12; all three independent repeated prompts returned4. With verbose trace diagnostics enabled, load took190.5s, API requests8.49/17.34s and repeats8.76–8.90s. This is text-only fixed-context acceptance, not universal Qwen or cross-model certification.

Actual Gradio package UI acceptance also passed; see `FXB_EX_UI_VALIDATION.json`. Load212.7s, answers4→12 with history, independent repeated answers4, request times3.48–3.58s after load (follow-up6.94s), and unload succeeded. The UI uses a separate worker process for reliable device release and a per-client `gr.State` for model history; Gradio6 display messages are not passed directly as model input. The first UI trial exposed that display/history-format mismatch; separating state fixed it, and the full UI test was rerun. UI remains available on server127.0.0.1:7861 with no model resident after testing. Mac tunnel: `ssh -F /Users/dong-jin/Documents/GitHub/furiosa/.ssh/config -N -L 7861:127.0.0.1:7861 furiosa-npu`.
