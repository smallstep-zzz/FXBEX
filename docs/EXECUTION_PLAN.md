# Shared execution plan — 0.8.1 source

`build_model(torch_module,...)` delegates to furyNo5 0.5.0; fxbX does not
recapture the model, generate TCL or compile during loading. String/path models
retain the official registered build path. Explicit legacy providers retain
their single-EDF route, rejecting constraints they cannot enforce.

```python
from fxbx import build_model, GraphFXB
build_model(model,'./model.fxb',example_inputs=inputs,concurrency=4)
with GraphFXB('./model.fxb',weights=model.state_dict()) as session:
    outputs = session.predict(inputs)
```

The archive contains a real v2 manifest, genuine EDFs and hash-bound
`tclmachine.execution-plan.v1` metadata. Ordered tensor IDs bind external weights,
intermediates and functional zero-initialized state. `GraphFXB` dispatches by plan
schema, never model name; `ExecutionFXB` is also public. The loader needs only
fxbX, matching SDK/driver, archive and external weights. It does not synthesize
Furiosa-LLM architecture registration, chat/streaming scheduling or KV paging.

Loading validates hashes, exact SDK, ordered opaque ABI, fusion8/Single, dataflow
and conservative logical memory. Last-use intermediates are freed; completion
precedes state commit. Reset/visible state copies are included in admission.
A failed kernel poisons the session; cleanup preserves its primary exception.
Pure, non-mutating EDF inputs remain a producer obligation: ABI text cannot prove
purity. Lookup IDs have explicit CPU-boundary range checks; internal computed IDs
without proved bounds remain blocked. This is validation, not CPU embedding.

## Server5 evidence — 2026-11-09

SDK2026.4.0, fusion8/Single. Small residual MLP, functional cell, repeated-shape
layers, ReLU special values, biased/unbiased batched MLPs and BMM passed CPU
comparison and ten repeats. Each profiled step had a matching official NPU Task;
CPU model routes were zero. Changed inputs, state/reset, NaN/signed-zero and EDF
reuse passed. Four archives reloaded in another process with producer imports
blocked and compiler invocation forbidden. Tiny median host request times were
0.388–0.511ms including H2D/dispatch/sync/D2H, excluding trace collection.
These checks do not establish full-model performance.

Nine generic graph cases /16 unique EDFs /22 execution steps were confirmed by
the official NPU trace, with zero unknown steps. The int32 embedding+projection
checks passed for1D/2D IDs with ten changed-input calls; invalid negative/OOB IDs
were rejected before dispatch and the session remained healthy.

Actual Qwen9B capture/binding now passes432 input bindings after generic wrapper
and local-copy normalization. Source context262144 and config hash are preserved.
Full native export remains blocked:29244 unresolved call nodes group into54 op
kinds in the pre-lookup-support snapshot. Its first gap, embedding, now has a
shared small-kernel NPU qualification; the whole9B graph was not requalified.
Capture is not compilation/NPU/full-context acceptance.

hf-librarian prepared YOLO and SD-Turbo. CPU capture covered detection895nodes,
text encoder866, UNet1974 and VAE415. Conv/norm/activation/layout/attention/upsample
families need more shared EDF rules. Legacy working model runners remain separate
and are not evidence for this new full-model native-FXB route.

Raw traces/features remain in `/root/fxbx_tclmachine_test/` onserver5; small result
JSONs and replay scripts accompany this document. This is source-only work, not a
new PyPI publication.

## Legacy engine shadowing

An editable checkout in a `--system-site-packages` environment can resolve
`fxbx` from the new checkout but `fxbex` from an older system installation.
The public import now checks that its engine is the bundled sibling and fails
with both paths and repair advice. It does not rewrite global import paths.
In the matching SDK environment, install the checkout normally:
`python -m pip install --force-reinstall --no-deps /path/to/fxbX` (without `-e`).
The server5 isolated wheel installation passed the real2.03GB checkpoint lookup;
SDK packages were unchanged. Metadata/mmap load time is not disk-read throughput.

## 0.8.3 source: inputless prepared values

Exact EDF ABI may have zero inputs; outputs must still be nonempty. The producer
may label a kernel `execution_kind: static_value_preparation` only with zero
inputs. This is not NPU-compute evidence. Other kernels default to `npu_compute`
and retain the normal numerical/official-trace gates. Feature discovery is
`ExecutionFXB.supports_zero_input_kernels`; older consumers fail before compile.

An inputless EDF can return a resident compiler constant. Logical memory admission
therefore conservatively counts these outputs as persistent, and NPU outputs are
cloned before exposure so caller mutation cannot change future results. CPU output
copies follow the usual transport path. Source tests include exact ABI rejection
and mutation isolation. SDK2026.4 int32/int64 arange packages were exported,
reloaded and numerically checked over five calls. This certifies prepared values,
not a new integer arithmetic engine or whole-model support. 0.8.3 source is not
claimed as a PyPI deployment by these checks.

## Source update: guarded frozen-buffer specialization

`tclmachine.execution-plan.v2` adds nonempty `frozen_buffer_guards`. Each entry
binds a declared buffer target to its exact shape, canonical dtype and SHA256.
Hash framing is dtype ASCII + newline + compact JSON shape + newline + contiguous
raw tensor bytes. Only frozen plain CPU buffers are eligible (64KiB each,1MiB
total); parameters and runtime input values are not static buffer constants.

The producer must check `ExecutionFXB.supports_frozen_buffer_guards` before
specializing. The consumer validates dict or streamed weights before runnable
deserialization and NPU allocation. A changed value fails closed. Unguarded
programs stay schema v1; v1 with guards and v2 without guards are invalid. Older
consumers reject v2, so use a matching consumer source, not an assumed version.
This update has not been published to PyPI in this validation checkpoint.

Server5 SDK2026.4 qualified guarded embedding and scalar preparation over five
calls each, including rejection of a changed buffer. MiniLM BF16 encoder[1,16]
then reloaded an actual native FXB in a producer-free fresh process: eight calls,
zero compiler attempts/SDK CPU routes, all330 steps with NPU Task evidence.
There are31 distinct compute EDFs and2 prepared-value EDFs. CPU BF16 hidden-state
cosine was0.99996543; observed unprofiled warm median12.95ms is not an isolated
performance comparison. Other sequence lengths and sentence pooling remain
unverified; this is not arbitrary-model or full-context acceptance.
