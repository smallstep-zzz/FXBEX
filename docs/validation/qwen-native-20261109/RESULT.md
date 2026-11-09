# Qwen3.5-9B native FXB check — 2026-11-09

**Qwen native FXB generation failed; no Qwen FXB inference or full-context acceptance.**

Actual checkpoint: RedHatAI/Qwen3.5-9B-FP8-dynamic, revision
790f0576d2d77dd5322aa0603a470bd9e3a3d1f6. Original text context is 262144.
The source config hash was unchanged. FP8 weights were prepared as BF16 through
the existing FHA loader; native FP8 compute is not claimed.

- Official build preflight with max-model-len262144 rejected the unregistered
  Qwen3_5ForConditionalGeneration architecture before compilation.
- Actual 9B model loading:77.06s; Torch export:25511 nodes/25077 calls.
- Custom native export failed with UNLIFTED_GRAPH: get_attr parameters/buffers
  must remain lifted. Total99.85s. This is the first observed producer binding
  failure, before NPU execution.
- The structural probe used sequence16, not a 262144-token inference. Preserving
  original config capacity does not validate full-length execution.
- Beyond this binding failure, the current fallback handles a single mm/linear,
  not the full multi-op graph or state scheduler. Fixing one guard alone will
  not complete native Qwen export. Working legacy FXBEX execution is separate.

Custom export now rejects context/bucket/compiler options instead of silently
ignoring them. Registered-model options retain their official builder route.
Six model-export unit tests passed.

Positive control: a new genuine Linear native FXB (205678 bytes) was generated,
its manifest/EDF/provenance and ordered user/weight/bias bindings opened, then
a separate NPU process returned shape[4,6], CPU-oracle max absolute error0.
Build0.842s; load+oracle+call0.060s. This is not Qwen acceptance.

Server source, errors and receipts:
`/root/fxbx_tclmachine_test/qwen-native-validation/`.
No source config rename, fake architecture, reduced-context success claim or
Runnable-as-EDF substitution. Full-length prefill/decode/KV/page-boundary and
memory tests remain required after a real full-model FXB exists.
