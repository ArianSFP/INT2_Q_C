# TACTIC actual coarse N18 v6 independent result auditor v1

Date: 2026-09-02

Status: minimal post-failure schema repair awaiting a rerun against the same
independently recorded external pin bundle. The sealed v0 package failed with
`decoder original score: mapping fields`. Diagnosis opened the failed log and
the score metadata in authenticated `RESULT.json` and `DECODER_RECEIPT.json`;
their numerical metadata was therefore visible. No `COARSE.bin` or BF16 input
payload was opened while repairing and freezing v1, and CUDA/CuPy was not
initialized. `UNRESOLVED_EXTERNAL_PINS.json` remains intentionally unusable.

## Exact v0 defect and v1 repair

The authenticated v6 producer deliberately emits
`scorer_uses_exact_encoder_input_bytes: true` in its producer-facing score
mapping. The v0 independent recomputation correctly scored the pinned BF16
bytes, then constructed its comparison view by removing redundant raw-MSE and
weight-count fields. It accidentally omitted the producer's input-byte
attestation and demanded exact mapping-field equality, causing a false
failure before any numerical comparison.

V1 changes only that comparison adapter: the expected producer-facing view
now requires the attestation to be present and exactly `true`. It continues to
reject a missing, false, or additional field. The producer, publication,
external pins, numerical decoder, and independent score path are unchanged.

The failed evidence remains on the RunPod and is pinned in
`PRIOR_FAILURE_RECORD.json`. This successor is not blind: it was selected
after observing the v0 failure and score schema. Its numerical replay is still
independent because producer JSON remains comparison-only.

## What a PASS proves

The auditor retains and authenticates the complete v6 and v4 source closures,
runtime lock, source-free smoke, identity-free input manifest, three exact
BF16 inputs, separately pinned numerical dependencies, all eight publication
members, and terminal completion seals. Only after those bytes are retained
does it import NumPy and execute a separately authenticated CPU polar core.

It independently implements the `TACN18C4` packet parser and canonical
builder, causal six-level SC decode, little-endian I32 inverse Hadamard,
original-BF16 FP64 scoring, rational physical-rate calculation, completion
verification, and traffic ledgers. It requires literal canonical re-encoding
of all 18 reservoirs and recomputes pooled SSE, raw MSE, relative MSE, exact
bytes, exact `307/128` bpw, and a coarse-only diagnostic F. Producer JSON is
comparison-only.

## What a PASS does not prove

A PASS authenticates one externally pinned 768x2048 three-role byte set. Shape
does not prove checkpoint provenance. It does not implement the 384-bit fine
stage, establish a final 2.5-bpw codec, establish arbitrary-shape tail rate,
or establish universal SwiGLU-MoE performance.

The auditor executes exactly one external `COARSE.bin` file read and reports
that audit I/O separately from host-memory and scratch lower bounds. Page
counts are projections, not an executed routed-inference trace. Accelerator
HBM is unmeasured, and the receipt has no `<2x` inference-HBM authority.

Static completed files cannot prove the historical `renameat2` syscall order.
The audit instead requires independently recorded hashes for all members,
exact directory membership, canonical completion rows, both internal seals,
retained name/inode/byte binding, and a completion file not observably older
than any ordinary member. This limitation is explicit in the receipt.

## Invocation after external pinning

```text
/workspace/int2-cupy-venv/bin/python -I -B \
  research/tactic_actual_coarse_n18_v6_result_auditor_v1/result_auditor.py \
  --authorization AUDIT_EXACT_TACTIC_ACTUAL_COARSE_N18_V6_QWEN_RESULT_V1 \
  --expected-auditor-source-manifest-sha256 <SOURCE_MANIFEST SHA-256> \
  --external-pins /absolute/canonical-external-pins.json \
  --expected-external-pins-sha256 <independently recorded SHA-256>
```

The verifier is read-only and prints one compact, internally sealed receipt to
stdout. The exact CPU decode is expensive.

## Source-only checks

```text
python -I -B test_source_only.py
python -I -B verify_source.py \
  --package /absolute/research/tactic_actual_coarse_n18_v6_result_auditor_v1
```

The hostile tests use only synthetic bytes and standard-library code. They
cover strict JSON, unresolved/weakened pins, packet/header/payload tampering,
canonical re-encoding, I32 bounds, exact rational rate, completion sealing,
nonpromotion, Qwen geometry, and traffic-accounting boundaries.
