# Shared execution plan — 0.8.0 source

`build_model(torch_module,...)` delegates to TCLMachine0.5.0; fxbX does not
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
