# Automatic model export: two explicit routes

This change implements orchestration and offline-tested binding/plan code.
No additional server/NPU experiments were performed, as requested.

```python
from fxbex.model_export import build_model

report = build_model('/path/to/HF-model-assets', './model.fxb', concurrency=4)
```

A model ID/local asset directory invokes the public official architecture-aware
builder, validates its output, adds FXBEX producer metadata and atomically writes
the native FXB. Unsupported architectures retain the exact registry error; they
are not renamed. Weight assets remain separate. This new automatic entrypoint
has CPU/mock coverage, not a new SDK build/inference acceptance claim.

```python
report = build_model(
    torch_model, './custom.fxb', example_inputs=(example_tensor,),
    source_metadata={'architecture': 'ActualModelArchitecture'},
    contract_provider=authentic_native_contract_provider,
)
```

The custom route exports a static functional graph with parameter/buffer tensors
lifted as ordered runtime arguments. TCLMachine invokes the public native
compiler, requires one actual EDF and records binding/ABI metadata. Checkpoints
are not serialized into the package; varying external weights still requires
runtime validation to establish that compilation did not specialize values.
Noncontiguous/dynamic/state-mutating/multi-EDF cases stop with explicit errors;
there is no guessed DAG or silent CPU model fallback.

FXBEX joins the result to a real native model contract through `NativeContractProvider`.
That provider must bind the target architecture/configuration, exact EDF reference,
bucket, provenance and device setup. The default provider explicitly reports
`EXPORT_BRIDGE_GAP`: inspected public APIs do not supply a custom-graph native
fingerprint/registration constructor. A provider stub that raises this error is
not completion of arbitrary-model native FXB generation.

Completed EDFs and `edf-contract.json` survive packaging failures. Use
`resume_graph_build(work_dir, output, contract_provider=...)` to package a corrected
contract without recompiling. Output files are never overwritten. The generic
graph plan, external-weight bindings and explicit official EDF loader consumer
are described in `GRAPH_FXB.md`; automatic Furiosa-LLM scheduling is separate.

At present, fully automatic new custom-architecture contract creation remains
unresolved. The implementation must not advertise universal model export or
official LLM compatibility based solely on offline tests or archive generation.
