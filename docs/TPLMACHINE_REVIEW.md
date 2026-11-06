# Measured optimization handoff

FHA owns feature extraction, candidate translation, the official compiler and NPU
acceptance. TPLMachine owns bounded search and scope-matched ranking. FXBEX owns
verified packaging, fixed-plan loading and execution receipts. A simulation or a
microbenchmark winner cannot activate different executables in an existing package.
Changed strides require new captured requests and a new immutable delivery.

Include the small `optimization_selection.json` and independent model acceptance
summary as ordinary integrity-mapped package files at build time. They must identify
source/code, SDK/compiler, hardware/topology, shape/dtype, timing/completion scope,
reference checks, trace path/hash and repeat measurements. Prediction, compiler
estimate and hardware measurement stay separate. `deployable:false` feature results
are planning records, not model support. Replay misses must never trigger compilation.

## Review of the supplied report against 535e1cc

Implemented first: verified-root preflight shared by validate and Session before
device upload, static DAG input dtype/shape checks, preserving integer/bool outputs
(removed float coercion), common generate execution receipts, and invalidating a
session after execution failures. Known pre-dispatch input errors remain retryable;
arbitrary adapter exceptions conservatively require reload because exception type
does not prove that model state was untouched. This is not a device-failure verdict. A native
segfault still requires worker/process isolation and cannot be caught in Python.
Compatibility/integrity checks remain mandatory. Full verification precedes imports
of bundled adapter code. Preflight currently follows extraction; it does not save
archive extraction cost for incompatible environments.

Existing single-pass hash-during-extraction stays intact. Internal RunnableDAG
weight hashes and the hug constructor's extra full validation are deliberately
retained until a verified immutable-handle contract exists. A path or a disk JSON
receipt is not sufficient proof for skipping hashes. No unsafe persistent cache
or concurrent SDK uploads were introduced.

Deferred behind independent hardware tests: mmap weight sources, byte-bounded
upload queues, per-region allocator admission, event-based release, chunk prefill,
multi-session KV ownership, residency and native Rust execution. The current
adapter has no validated multi-token prefill kernel; replacing its loop alone is
incorrect. These changes are separate from selecting a padded input layout.

Public upstream reference: https://github.com/furiosa-ai/furiosa-opt-models and
https://github.com/furiosa-ai/furiosa-opt. Their native runtimes are not a drop-in
replacement for SDK 2026.4 torch Runnable. Use optional, version-pinned adapters
and official compiler outputs; do not install a preview toolchain automatically.

SMI zero remains inconclusive. Rust Device::usage describes that runtime's allocator
and memory regions; it does not establish the allocation size of an unrelated torch
process. No allocator bridge or corrected physical HBM counter has been verified.
Keep logical tensor estimates and physical unknown distinct.

Scheduler integration uses settings.npu_lock(): an inherited file descriptor must
match the canonical lock's device/inode before adoption. Closing a duplicate never
unlocks the parent's lease. Standalone callers retain the ordinary lock path;
older cores reject scheduler adoption instead of silently running without a lease.
This permits CPU prepare -> exclusive NPU -> CPU feedback without nested-lock
self-deadlock. Native execution remains a separate backend contract.

Optional optimization metadata binds advisory selection and independent acceptance
to integrity-mapped files, execution-plan and compatibility hashes. It does not
patch schedules at runtime, declare current requests NPU-proven, or promote a
feature speedup into model support. The current Qwen candidate's independent
reload segfault blocks promotion; published legacy packages remain unchanged.
These additions are source-level changes, not a claim of a new PyPI release.
