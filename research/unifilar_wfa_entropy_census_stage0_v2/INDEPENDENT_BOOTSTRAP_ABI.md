# UWFA-SC v2 independent bootstrap ABI

The producer cannot authenticate itself. A separate audit package must contain
an immutable dispatcher whose own manifest is independently pinned and whose
source hard-codes:

- the exact SHA-256 of this package's `SOURCE_MANIFEST.json`; and
- the exact SHA-256 of the independent source-review receipt for that manifest.

A public token or internally self-sealed review is not authority.

## Required dispatcher order

The dispatcher is launched with isolated Python `-I -B` and must:

1. reject a wrong external authorization before output or dynamic-input access;
2. walk the package/review/source input paths component-by-component with
   descriptor-relative no-follow opens;
3. retain every directory and file descriptor;
4. require the externally pinned manifest/review digests;
5. reject undeclared package members;
6. read every source member exactly once, authenticate size/hash, and retain
   both bytes and descriptors;
7. compile/exec `stage0_census.py` and every sibling module from those buffered
   bytes—never `spec_from_file_location` or a pathname import;
8. open, authenticate, parse, and replay the source artifact before importing
   CuPy or loading `cupy_backend.py` from its buffered source;
9. call `gpu_preflight_all_150`; abort before fitting if it fails;
10. call `source_phase`; open no Gaussian payload unless
    `controls_may_be_opened` is literal `true`;
11. if authorized, open/authenticate all eight controls, then call
    `controls_phase`;
12. write returned bytes through one retained output directory descriptor,
    fsync every file and directory boundary, and create `COMPLETE.json`
    exclusively last; and
13. verify every held identity before completion.

The reference open/snapshot primitives in `dispatcher_contract.py` are test
material. Calling that file directly is deliberately blocked because code
inside the producer closure cannot be its own independent authority.

## Snapshot module ABI

The dispatcher supplies modules compiled from authenticated bytes:

- `uwfa_common.py`
- `protocol.py`
- `container_codec.py`
- `strata_sc_adapter.py`
- `cupy_backend.py`
- `stage0_census.py`
- universal frozen STRATA format and independent-decoder modules pinned by the
  external authorization record.

Construct `StrataSCAdapter(common=..., np=..., frozen_auditor=...,
strata_common=..., device="cupy")` only after the baseline replay gate.

Construct `BoundEvidence` with eight exact SHA-256 values. Call:

```text
source_phase(
    common, protocol, container_codec, adapter, backend,
    artifact_bytes, score_receipt_bytes, bindings, gpu_preflight
)
```

The returned `_container` and `_identity_framing_container` bytes are output
members. Other underscore-prefixed in-memory objects are not serialized.

If and only if the source result authorizes controls, prepare eight exact
bundles in frozen seed order and call `controls_phase`. Every bundle contains
literal artifact/score bytes, an externally authenticated provenance binding,
and its own `BoundEvidence`.

## Required output members

At minimum:

- `RUN_STATE.json` (created first, incomplete);
- `uwfa_sc_v2.container`;
- `same_framing_baseline.container`;
- `result.json` without in-memory objects;
- `environment.json`;
- `page_trace.json`;
- exact source/review/authorization binding record;
- any control results, only when authorized; and
- `COMPLETE.json` created exclusively last.

No positive scientific status exists until a separate fresh-process result
auditor, not importing producer code, reparses the container, regenerates every
SC context, reconstructs and rescores original weights, and independently
recomputes actual rate, F, and cold pages.
