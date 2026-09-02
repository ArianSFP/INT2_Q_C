# TACTIC actual coarse N18 v6 independent result auditor v0

Date: 2026-09-02

Status: source-only frozen package awaiting an independently recorded pin
bundle for one terminal real-result directory. This package was built without
opening the live v6 result, any Qwen/BF16 input, CUDA, CuPy, or network
payload. A copy of these source files was exercised with standard-library
Python under a fresh RunPod `/tmp` path; it did not open or enumerate the live
result, input manifest, or model paths. `UNRESOLVED_EXTERNAL_PINS.json` is
intentionally unusable.

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
  research/tactic_actual_coarse_n18_v6_result_auditor_v0/result_auditor.py \
  --authorization AUDIT_EXACT_TACTIC_ACTUAL_COARSE_N18_V6_QWEN_RESULT_V0 \
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
  --package /absolute/research/tactic_actual_coarse_n18_v6_result_auditor_v0
```

The hostile tests use only synthetic bytes and standard-library code. They
cover strict JSON, unresolved/weakened pins, packet/header/payload tampering,
canonical re-encoding, I32 bounds, exact rational rate, completion sealing,
nonpromotion, Qwen geometry, and traffic-accounting boundaries.
