# Generic graph assembly and explicit EDF execution

`fxbex.graph_fxb.export_graph` takes the report returned by
`tplmachine.edf_export.EDFExport.write`, target model metadata and an authentic
`NativeContractProvider`. It automatically stages the native manifest and EDF,
then calls the existing native writer. Ordered user/parameter/buffer bindings,
output metadata, opaque public EDF ABI, actual SDK versions, source graph
identity and compiler configuration are saved in the producer `graph_plan`.
Weights remain external. Manifest, model metadata, plan and EDF hashes bind those
records together; these hashes detect changes and do not authenticate a producer.

```python
from fxbex.graph_fxb import export_graph, GraphFXB

# edf_report comes from export_edf(...).write(...).
# provider is trusted host code backed by a genuine native contract producer.
report = export_graph(edf_report, 'model.fxb',
                      model_metadata=model_metadata,
                      contract_provider=provider)

with GraphFXB('model.fxb', weights=model.state_dict()) as loaded:
    flat_tensor_outputs = loaded.predict(example_inputs)
```

The provider's `provide(*, model_metadata, edf_export)` returns a mapping with:

- `manifest`: genuine native FXB v2 metadata registering exactly this single EDF.
- `kernel_name`, `bucket`, `member`: its exact native registration.
- `model_metadata_sha256`: SHA256 of the target metadata encoded with sorted
  JSON keys, compact separators and finite numbers.
- `source`: provider provenance.
- `device_config`: actual compiled configuration, currently supported only as
  `{'fusion': 8, 'mesh': 'single'}` by the explicit consumer.

The real model architecture must match the native fingerprint. Provider code is
trusted: a hash assertion alone cannot establish that SDK metadata is authentic.
`ExistingNativeContract` is a convenience for an already authentic contract,
not a way to relabel another model. The default is now `StaticGraphContractProvider`:
it binds the real architecture, graph/EDF hashes, ABI and recorded compiler/SDK
configuration into an fxbX static-graph fingerprint and native v2 manifest.
Official show/inspect and actual RNGD execution passed for the small qualified
cases; see [STATIC_GRAPH_BRIDGE.md](STATIC_GRAPH_BRIDGE.md). This custom fingerprint
does not assert official LLM-cache compatibility. The old explicit
`OfficialNativeContractProvider` still reports the missing official LLM model
registration API. Registered architectures retain the official build route.

`GraphFXB` explicitly selects the saved manifest member, verifies hashes and
bindings, loads it through `furiosa.torch.ir.Runnable.from_serialized_edf`, and
compares the public `input_dram_shapes`/`output_dram_shapes` representations
exactly before loading the device program. Topology is configured before EDF
preprocessing, because that preprocessing records fusion. It also checks actual
`expected_num_fusion`, pinned SDK versions, tensor shape/dtype/stride, and required
external parameters/buffers. Missing device metadata fails before allocation.
The loader uses the shared NPU lock, disables CPU model fallback, uploads weights
once and waits for public stream completion. Errors invalidate the loaded
session; cleanup releases its program, resident weights and lock.

The consumer supports one static, stateless graph, positional or named dense
tensor inputs and flat tensor outputs. Original output tree metadata is retained
for an orchestration layer to reconstruct. Mutation, dynamic shapes, multi-EDF
graphs, other device configurations and unsupported input kinds fail explicitly.
This consumer is distinct from Furiosa-LLM's architecture dispatch.

Opaque ABI equality proves that the reader recovered the recorded EDF interface;
it does not parse opaque SDK representations into logical tensor metadata.
`abi.logical_to_physical_metadata` remains `unknown` unless the EDF producer used
an explicit verified reader. `validation.external_weight_variation` remains
`not_tested` without a separate execution test. These limits are retained in the
plan, as are native writer runtime/NPU gates. The initial module was offline-only;
subsequent source 0.7.1 server5 verification is in `STATIC_GRAPH_BRIDGE.md`. Those
small static cases do not establish arbitrary stateful model dispatch.

Offline checks:

```bash
python -m unittest discover -s tests -p test_graph_fxb.py -v
```

Tests use deliberately fake EDF bytes and mocked SDK/tensors to verify binding
order, corruption rejection, ABI checks before device loading and cleanup.
They do not constitute numerical or NPU acceptance evidence.
