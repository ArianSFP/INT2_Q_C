# Independent runtime-receipt audit: lossy-tail v8

Status: **PASS_V8_INDEPENDENT_RUNTIME_AUDIT**.

This package independently authenticates and audits the exact source-free
runtime receipt copied from
`/var/tmp/int2_lossy_tail_v8_runtime_evidence_6c5f5cd0/calibration_v1/runtime_receipt.json`.
The remote and copied file are byte-identical: 82,235 bytes, SHA-256
`45862549f34530964c4f8f7a4134228ccf036a8de3534f23e63e07acde7985b3`.

The verifier binds the exact frozen v8 stage and independently passed source
audit, rejects duplicate JSON keys and nonfinite values, recomputes the
runtime receipt's canonical internal seal and probe aggregate, and checks:

- the exact pinned 14-field runtime tuple;
- 48 ordered `(replica, ordinal)` RNG cells and independently recomputed seeds;
- two ordered affine rows for every cell;
- five independently reconstructed stable-order fixtures;
- SHA-256 and canonical `float.hex()` syntax, uniqueness, and affine-moment consistency;
- all six memory-release fields for all 53 probe rows; and
- exact stage/contract/invocation bindings and the source-free access ledger.

The evidence pass contains 3,243 independent assertions. The audit itself
opened no model/Qwen/source payload, validation data, or production result,
did not import CuPy or initialize CUDA, and submitted no GPU job.

This PASS is runtime evidence required by the frozen authorization contract.
It is not production authority, does not open a payload, and does not make a
compression-performance claim. A separate external one-shot production
authorization is still required.

## Verify

From this directory on a machine with Python 3.9 or newer:

```text
python3 -B -I verify_runtime_audit.py
```

The verifier uses only the Python standard library. It requires the exact
frozen producer and independent source-audit directories to remain adjacent
to this directory.
