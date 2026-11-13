# Automatic static-graph native FXB bridge

Source version **0.7.1** adds an automatic native FXB contract for a qualified
stateless graph. The previously published PyPI 0.7.0 does not include this change;
use this source checkout until a new PyPI release is explicitly published.

```python
import torch
from fxbx import build_model, GraphFXB

model = torch.nn.Linear(8, 6).bfloat16().eval().requires_grad_(False)
x = torch.randn(4, 8, dtype=torch.bfloat16)
build_model(model, './linear.fxb', example_inputs=(x,))  # new output path

# A later process needs the .fxb and matching external state_dict, not a recapture.
with GraphFXB('./linear.fxb', weights=model.state_dict()) as loaded:
    y, = loaded.predict((x,))
```

The original weights stay separate, as with native FXB. Loading never invokes
the model compiler. `GraphFXB` is the explicit static-graph consumer; `LLM` and
`furiosa-llm serve` still require their own architecture implementation and state
scheduler. A native archive is not an automatic LLM-model registry extension.

## Ownership of the execution connection

1. Torch export records actual graph, user inputs, parameters, buffers and output
   tree. FuryNo5 binds these without inspecting checkpoint binaries.
2. When the installed SDK lacks `native_common.compiler.compile`, FuryNo5
   uses its existing finite public TCL rules and installed `furiosa-tcc` to emit
   a **genuine EDF**. Current fallback scope is one static contiguous BF16 2D
   `mm` or `linear`, including optional bias. Multi-op graphs, unhandled dtypes,
   mutation, aliases and unmapped compiler options fail explicitly.
3. Kernel argument order can differ from Torch's lifted signature. The exporter
   records the actual TCL operand order and corresponding state_dict names.
4. fxbX constructs native v2 metadata from the real architecture. Its graph
   fingerprint binds graph/EDF hashes, ABI, source, SDK/compiler versions, config
   and fusion8/Single. `forward`, EDF member and the one fixed-shape bucket are
   registered automatically. No Qwen/Solar identity or registry entry is borrowed.
5. Official `fxb show`/`inspect` validate the bundle. The CLI can live in another
   SDK Python environment. `GraphFXB` verifies hashes/ABI, configures topology
   **before EDF preprocessing**, uploads external weights, then invokes the exact
   saved EDF and synchronizes the public stream.

The native tokenwise bucket is a fixed-shape container entry whose input_size is
the first user tensor's leading dimension. It is not an LLM admission policy.
Do not auto-register these custom graph bundles in the official LLM cache or
claim `fxb check`/automatic LLM dispatch compatibility. Existing authentic native
contract providers remain supported for their separately verified consumers.

## Why a loaded FXBEX object cannot simply supply the missing registration

An executed NPU program does not create the model's host control flow, KV state,
tokenizer/scheduler or architecture implementation. In the installed SDK,
`Runnable.serialize` is reloadable Runnable state. Its public `binary` tensor
was **rejected** by the official EDF deserializer; it is not an EDF export API.
Renaming it or filling metadata after execution would lose that distinction.
Capture the binding at the producer boundary and preserve the genuine compiler
output instead. A retained successful EDF can use `resume_graph_build` without
recompiling; an arbitrary legacy `.fxbex` snapshot is not automatically convertible.

## REA and existing Furiosa tools

REA 4.0.1 inspected the isolated small test FXB: its ZIP resources and EDF hash
matched the experiment. This is static archive evidence, not a compiler
decompilation, runtime observation or proof of FXB semantics. Server5 lacks the
Node prerequisites, so that small inspection used existing local Node; all
compilation and NPU experiments ran on server5. No vendor binary was patched.

`cargo-furiosa-opt 0.10.0` is installed (the standalone name `furiosa-opt` is not).
It compiles Rust device crates and can dump schedule/IR/vISA/summary. Its runtime
uses `registry::Entry -> registry::load -> Function.launch` with image/topology
and argument binding. That `.bin` image is a different execution contract, not an
EDF or LLM FXB registration. For the current TCL path we reuse the installed
`furiosa-tcc` rather than replacing its optimizer or relabelling Rust images.
Rust schedule output remains useful for future Rust-kernel diagnosis, not for
inventing a missing LLM host implementation.

Official references: [FXB](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html),
[optimizer](https://developer.furiosa.ai/furiosa-opt/book/tools/kernel-optimizer.html),
[Rust runtime](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-rt/README.md).

## Server5 verification, 2026-11-09 KST

| Case | Reader gates | Numerical / repeats | Official trace | Warm median |
|---|---|---|---|---:|
| Linear bias, BF16 4×8→6 | show+inspect passed | exact CPU output, changed input, two external weight sets, 10 each | 2 Task events per weight set, CPU routes0 | 0.408 / 0.371 ms |
| mm BF16 4×8 @ 8×6 | passed | exact CPU output, 10 calls | 2 Task events, CPU routes0 | 0.540 ms |
| Linear without bias, BF16 4×8→6 | passed | exact CPU output, 10 calls | 2 Task events, CPU routes0 | 0.411 ms |

Times include H2D and completed D2H, exclude build/queue/first profiled call; tiny
single-process correctness runs are not model-serving performance benchmarks.
The extra cases forbid the public compiler entrypoint during artifact load/run.
The automatic `build_model` biased Linear path was also loaded in a new process.
These tests do not convert the full Qwen3.5 or SDXL Python orchestration to native
LLM FXB. Source/stateful/multi-graph support remains a separate development gate.

Receipts, source and raw traces are on server5 under
`/root/fxbx_tclmachine_test/native_bridge_20261109/`. Compact results are in
`docs/validation/static-graph-20261109/`; large artifacts/traces stay out of Git.
