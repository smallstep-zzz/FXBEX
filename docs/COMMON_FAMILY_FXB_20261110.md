# Native FXB common-family validation, 2026-11-10/11

**fxbX generated and executed three actual native FXBs through furyNo5's shared rules. Official readers, independent reloads and frozen-buffer rejection passed. This is small-feature acceptance, not arbitrary whole-model support.**

The existing fxbX consumer at commit `f4025ef` needed no runtime code change.
furyNo5 owns the reusable lowering and execution contract; fxbX packages and
loads its genuine EDF/native manifest. SDK2026.4/Torch2.10, fusion8/Single,
CPU fallback disabled. No SDK upgrade, proprietary binary analysis or model-name
production branch was used.

| File | Contents | Verified result |
|---|---|---|
| `common-families.fxb` | Transpose packing, FP32 vector matmul, constant padding, triangles/masked_fill, eye preparation, alias binding | 580,765 bytes; official show/inspect pass;3 fresh loaders/14 calls;196 Task events;6 compute steps and1 preparation;bit mismatches0/CPU routes0 |
| `stepped-families.fxb` | Step3 read, functional copy/scatter, repeated static positions, real external bias weights | 363,978 bytes;official readers pass;independent6 calls plus2 reloads×4 calls bitwise;readback112 Tasks,all6 compute steps traced,CPU0;axis-iota is1 preparation step |
| `bounded-trig.fxb` | Proven finite frozen phase, range-reduced sin/cos, ordinary FP32 arithmetic | 230,771 bytes;official show/inspect pass;plan v2 with1 frozen-buffer guard;independent7-call replay plus2 reloads×4 calls;5/5 compute graphs,unknown0/CPU routes0;max error2.385e-7 |

For `bounded-trig.fxb`, changing the phase buffer without changing its shape is
rejected with `Frozen buffer hash mismatch: phase`, before NPU program creation,
prediction, compilation or subprocess execution. General unbounded sin/cos is
not enabled. Independent reloads prohibit producer imports and recompilation.

These are native **weightless** FXBs. Actual checkpoint weights/buffers are
separate, as required by this route. The common and stepped fixtures use real external
SafeTensorWeights rather than hiding values in a custom archive.

```python
from fxbx import GraphFXB, SafeTensorWeights

with GraphFXB("common-families.fxb",
              weights=SafeTensorWeights("common-families.fxb.verified/weights")) as model:
    outputs = model.predict((input_tensor,))
```

The common fixture input contract is FP32 `[8,8]`; the stepped fixture has two
FP32 `[1,16,32]` inputs and one `[3,1,16,32]` output. This example is not a general image/LLM
adapter. The corresponding public reproduction scripts and small evidence are
in furyNo5:

- `tests/hardware/probe_common_family_fxb.py`
- `tests/hardware/probe_bounded_trig_fxb.py`
- [Common-mechanism report](https://github.com/smallstep-zzz/furyNo5/blob/main/docs/COMMON_MECHANISM_REPAIR_20261110.md)
- [Small JSON receipts](https://github.com/smallstep-zzz/furyNo5/tree/main/docs/validation/common-mechanisms-20261110)

Remote artifacts remain under `/root/fxbx_tclmachine_test/` on servers25 and5.
Large traces, EDFs, weights and generated FXB binaries are excluded from Git.
An official-readable manifest and explicit GraphFXB execution do not register a
new architecture in Furiosa-LLM or create a dynamic KV scheduler. Whole HF-model
and final broad verification were excluded by the user's requested scope.

Two diagnostic harnesses initially assumed a successful build retained
`preflight.json`; the existing producer did not. Their failed logs were retained,
and the already-created FXBs were reused for readback. furyNo5 now retains
the admission-phase receipt before compilation on both normal and failing paths,
and exposes its path in the build report. This fixes diagnostics without changing
the compiled program or weakening execution gates.
