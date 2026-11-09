# Official state/cache reuse (2026-11-10)

Preserve the installed SDK 2026.4.0. Reuse its native model engine for registered
architectures. For custom Torch/EDF programs, adapt public ownership contracts;
do not reinterpret Rust HbmBuffer pointers as Torch tensors.

## Sources and compatibility

Public [furiosa-opt-models](https://github.com/furiosa-ai/furiosa-opt-models/tree/88989569875075d15c44e38b6883828072fb43ac)
provides `docs/develop/compose.md` (weights on Module, request state borrowed),
`crates/furiosa-opt-engine/src/decoder/cache.rs` (CacheSpec and exclusive lend),
and `registry.rs` (architecture implementations). Its compiler dependency pins
preview std revision `9481ae2db8c465b9c7008dcfa787721982789d3e`; server5 has
stable cargo-furiosa-opt 0.10.0 and no Python furiosa_opt_engine. That exact
preview backend was not installed or tested. This is not evidence that every
Rust API is incompatible. The public registry's Qwen3.5 entries are empty;
changing an architecture string does not create an implementation.

Decisions:

| Feature | Decision | Implementation |
|---|---|---|
| Registered model weights/KV/prefix scheduling | KEEP/direct reuse | official LLM through hug, original FXB |
| Weights separate from request state | ADOPT contract | named functional program, shared immutable tensors |
| Exclusive state borrowing | ADAPT | per-session lock and scoped RequestState |
| Rust cache allocator/paged KV ABI | DEFER | no demonstrated Torch/EDF ABI bridge |
| Architecture-name substitution | REJECT | explicit implementation required |

## Custom functional programs

```python
from fxbx import build_program, GraphFXB
# wrappers must share the same parameter objects and external target names
def entry(module, inputs):
    return dict(model=module, example_inputs=inputs,
                state_inputs={1: "cache"}, state_updates={"cache": 1})
build_program({"prefill": entry(prefill, prefill_inputs),
               "decode": entry(decode, decode_inputs)}, "program.fxb")
with GraphFXB("program.fxb", weights=prefill.state_dict(),
              max_request_states=2) as runtime:
    with runtime.new_state() as request:
        output = runtime.run("prefill", (x,), state=request)
        output = runtime.run("decode", (token,), state=request)
        runtime.reset(state=request)
```

Tensor state shapes/dtypes/layouts must match between entries. Incompatible
contracts fail before packaging. The FXB includes genuine EDFs and execution
metadata; **weights are separate**, passed explicitly. This native format is
not a replacement for the earlier portable archive containing weights.
Default state is allocated lazily; explicit clients do not allocate a redundant
default cache. Request admission uses a positive count limit (default 1) plus
conservative logical memory budgets. Queue wait is reported separately from
completed host request time; this is not pure kernel latency. State commits only
after device completion; execution errors poison the session. Close releases
its handles. No model-specific dispatch was added.

## Actual evidence on server5

- Official Qwen3-8B-FP8 FXB: load 5.146 s; two 2-token requests 0.039/0.028 s;
  292 correlated NPU Tasks; prefix cache queries 11, hits 5. Context 512 is a
  compatibility probe, not original-context validation. CPU model audit was
  not collected. No new download/config change. No throughput conclusion.
- `tests/hardware/probe_borrowed_state_program.py`: one FXB, two entries,
  four unique kernels, shared NPU state slot, two isolated clients; 14 interleaved
  calls with CPU reference, independent reset, unchanged resident weight objects.
  Four distinct execution steps have correlated official NPU Tasks, no CPU route.
  Final durable job completed in 7.369 s (includes build/validation).
- 38 focused fxbX unit tests and 3 producer tests pass. Producer tests mock
  compilation and are not additional NPU evidence.

Artifacts/raw traces remain under `/root/fxbx_tclmachine_test/official-cache-reuse/`
and `/root/fxbx_tclmachine_test/borrowed-state-program-v2/`. Job completion uses
`python -m tplmachine.research.jobs wait JOB`; disconnect does not stop workers.
The official probe intentionally is not repeated just to reformat its report.

## Remaining gates

Functional fixed-size tensor state is **not** paged attention, dynamic allocation,
full-context Qwen3.5, whole-model YOLO/diffusion, or Furiosa-LLM registry dispatch.
Those require admitted graph rules and explicit memory/KV execution contracts.
SMI zero does not prove absent allocation. Existing REA evidence is bounded
archive/resource/hash inspection, not recovered compiler ABI. Public sources
were sufficient here, so no new REA analysis was performed.

## Broader upstream reuse audit

The [FHA reuse audit](https://github.com/smallstep-zzz/Furiosa-Hug-Anything/blob/main/docs/FURIOSA_OPT_MODELS_REUSE_AUDIT.md)
compares the newer public source revision with these implementations. Installed
Furiosa-LLM reuse is distinct from direct use of the preview Rust engine.
Before a new workaround, use FHA's offline `python -m forge_fxbex.upstream`
lookup to locate registry refusals, unimplemented leaves and relevant functions;
source declarations never replace installed-SDK and actual NPU validation.
