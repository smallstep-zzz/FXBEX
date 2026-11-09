# Strict streaming checkpoint weights (source 0.8.2)

This adopts the public furiosa-opt-models Checkpoint ownership approach using
the installed SDK 2026.4.0 and safetensors' public APIs. It does not load the
preview Rust engine or share its memory ABI.

```bash
# From a checkout of this source release, in the existing SDK environment:
python -m pip install --no-deps .
# safetensors is optional; install fxbx[weights] when this release is published,
# or install the existing environment's compatible safetensors separately.
```

```python
from fxbx import GraphFXB, SafeTensorWeights

# A local .safetensors file or directory (including indexed shards).
weights = SafeTensorWeights("./checkpoint")
with GraphFXB("./program.fxb", weights=weights) as model:
    output = model.run("forward", (input_tensor,))
```

For an explicitly selected component or different target names:

```python
weights = SafeTensorWeights("./checkpoint", names={
    "projection.weight": "component.projection.weight",
    "projection.bias": "component.projection.bias",
})
```

Targets come from the producer's real execution plan. The mapping is data, not
a model-name exception. All selected targets must be bound. There is no guessed
prefix stripping, architecture substitution, implicit BF16 conversion or ignored
scale tensor. An FP8 checkpoint cannot satisfy a BF16 contract through this API.
Use the original validated adapter when conversion is required.

Construction and `validate` use safetensors headers only. Actual value access
happens one tensor at a time during device loading; the loader synchronizes
before dropping that CPU view. No full CPU nn.Module or CPU tensor dictionary
is created. Repeated plan references to one target share its uploaded tensor.
Integer buffers with value-range guards are additionally read before device
loading to enforce those guards. Existing dictionary-based loading is unchanged.

Duplicate tensor names, index/header disagreement, missing/extra targets and
incompatible dtype/shape/contiguous layout fail before device loading. File
identity checks detect ordinary replacement/modification; these are not content
hashes or a defense against a concurrent hostile writer. Keep checkpoint files
immutable while loading. The format's native FXB still stores EDF/plan metadata;
weights remain an explicit external input.

Currently supported by the native **execution-plan** consumer, which TCLMachine
produces. Legacy single-EDF graph packages need their existing tensor mapping;
passing this provider fails explicitly rather than materializing the whole file.
This API does not automatically replace legacy portable runners.

## Actual server5 evidence

`tests/hardware/probe_checkpoint_weights.py` reused the existing small
TCLMachine shared-linear native FXB, without recapture, compilation or producer
imports. Three replays, 21 calls, CPU reference checks and reset passed.
All four distinct steps had correlated official NPU Tasks; no CPU execution
route was observed in that test. Metadata refusal happened before SDK Runnable
device construction and with zero tensor value reads.

Metadata took 0.24–0.30 ms; first load 71.3 ms, subsequent loads 6.4–6.8 ms;
unprofiled completed host call median 0.485 ms. These are tiny-fixture timings,
not a comparison with the previous loader or a large-model speedup claim.
The test's CPU oracle and small checkpoint creation are intentionally separate
from the lazy loading route. Four provider unit tests and the existing 38 focused
runtime/API tests pass. Raw trace and outputs:
`/root/fxbx_tclmachine_test/checkpoint-weights-reuse/`.

Full-model peak RSS, real safetensors sharded checkpoint loading throughput and
quantized projection support remain separate validation gates. Per-tensor sync
bounds live CPU views but can add overhead; choose this route for bounded host
memory and explicit checkpoint binding, not an unmeasured latency guarantee.
