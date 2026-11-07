# Native weightless FXB preservation writer

`fxbex.native_fxb` writes an official-format ZIP from an **existing native FXB v2
manifest and its EDF kernels**. It copies the original `manifest.json` bytes and
only its `edfs/*.edf` references. Checkpoints, tokenizers, Python adapters and
Runnable files are excluded. The model assets remain separate, as required by
the [official FXB lifecycle documentation](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html).

```python
from fxbex.native_fxb import build_native, validate_native

report = build_native(
    '/path/to/extracted-native-artifact', '/path/to/new-model.fxb',
    producer_metadata={'source_native_artifact': 'recorded-source-identity'},
)
report = validate_native('/path/to/new-model.fxb')
```

The default requires the installed official `furiosa.llm_native.fxb.show` and
`inspect` readers to accept the staged archive before it is published. Inspection
parses the actual EDF signatures. No model compilation or NPU inference occurs.
An existing output is never overwritten. `verify_official=False` performs only
structural and ZIP CRC checks; its report explicitly leaves official reader
gates `not_run`. It is useful for offline inspection, not deployment acceptance.

Producer information is stored in a separate `fxbex.provenance.json` ZIP entry.
This is an **experimental FXBEX extension**, not an official manifest field or
fingerprint component. It records FXBEX's actual packaging version and operation;
caller-supplied metadata cannot replace those fields. Official reader acceptance
is reported separately from runtime loading. An official reader accepting an
extra ZIP entry does not establish a documented extension contract or successful
runtime support. The original architecture, fingerprint, compiler versions,
UUID and creation timestamp are preserved because they describe the compiled
kernels, not this archive operation.

The report distinguishes archive validation, official `show`, official `inspect`,
internal metadata reader acceptance, model compatibility, runtime loading and
NPU execution. This API leaves the last three `not_run`: those gates require a
target model and separate tests using the matching official runtime.

## Current custom-model boundary

On the inspected SDK 2026.4.0 installation, the public Python wrappers expose
`furiosa.llm_native.build.build_fxb` and the official FXB inspection/cache APIs.
They do not expose a custom-plan manifest writer, a fingerprint constructor or a
Runnable-to-EDF conversion API. The official documentation describes fingerprints
and example metadata, but does not publish a complete custom writer schema or a
producer extension mechanism. This writer therefore preserves a genuine native
contract instead of generating one from an arbitrary model config.

Passing an FXBEX manifest fails with `EXPORT_BRIDGE_GAP`. Repacking a native FXB
does not prove that new PyTorch/TCL graphs can run through Furiosa-LLM. That path
still needs genuine native EDFs with the expected kernel names, bucket semantics,
weight/state layout, fingerprint and runtime architecture implementation. An
existing `.fxbex` package remains usable through its existing runtime and is not
renamed or changed by this module.

No checkpoint, compiler-cache binary or proprietary binary was reverse
engineered. Format discovery used the official documentation, installed public
Python source/API docstrings, and JSON/archive metadata in an existing native
`/root/qwen3-8b-fp8.fxb`.

Run the offline contract tests with:

```bash
python -m unittest discover -s tests -p test_native_fxb.py -v
```

The unit fixture is deliberately not an executable EDF, so tests either use
structural-only validation or mock official reader acceptance. Real SDK evidence
must be recorded separately; unit tests are not NPU execution evidence.

## Recorded SDK check (2026-11-07)

[Native reader validation](NATIVE_FXB_VALIDATION.json) records the real SDK run:
169 genuine EDFs from the existing Qwen3-8B-FP8 native bundle were archived with
the internal producer entry. The original manifest bytes and every EDF's size
and ZIP CRC were preserved. Official `show` and `inspect` accepted the output.
[Compatibility validation](NATIVE_FXB_COMPATIBILITY.json) records a separate
official `cache_add` / `check` against `furiosa-ai/Qwen3-8B-FP8` at pinned revision
`7cf451930f6022204fcea071cfaeaf8aa22d570e`: the architecture fingerprint,
compiler and runtime revisions matched, and the official checker recommended
the bundle. The isolated compatibility cache was removed after the test.

These are two independently produced archives with preserved kernels and
different caller provenance; the JSON files identify their exact paths and
hashes. Runtime load and NPU execution are outside these reports. The remote
checkout's existing `fxbex.__version__` reported `0.5.0`, which the archive records
honestly; its `pyproject.toml` reports `0.5.1`. The writer did not change that
pre-existing version drift.
