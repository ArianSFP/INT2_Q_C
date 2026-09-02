# Global STRATA RM-order swap v2 authority

Date: 2026-09-02

Status: **frozen source-only repair; no payload or Qwen authority**.

This package is a narrow authority successor to frozen producer v1 source root
`980a5f1d272ca5ffc7b4d35e7c234a86994d135fcacaf0d47a8b3e00fc3d4f14`
and its independent nine-gap review source root
`1dfa55969b87543adbee785d72933f9ccb6f754eaade9e4e340a022c96c1afa8`.
It changes no RM selection rule and contains no weight payload, checkpoint
discovery code, Qwen table, or production decoder.

## What v2 repairs

1. Every package-root check performs `lstat()` and rejects a symlink before
   `resolve()`.  The standalone verifier follows the same order.
2. The current base encoder, BEC wrapper, and historical decoder are copied by
   exact hash into a fresh read-only closure before an integration import.
3. A physical decoder is copied by exact hash into a fresh read-only closure
   before execution; its source is rehashed after execution.
4. Production authority requires a successful independent decoder-audit
   receipt, audit-source root, manifest hash, and receipt hash as separate
   out-of-band parameters.  They cannot be supplied by the experiment
   commitment.
5. The authority-owned launcher instruments packet opens and byte intervals.
   It gives the decoder no source path, rejects explicit reads outside the
   request and packet, denies `os.open` and process/network escape, and derives
   read amplification without trusting a decoder trace.
6. Source identity, checkpoint/tensor identity, matched-control generation,
   model family, and cross-family provenance live only in a separately hash-
   pinned independent-auditor capability.  The experiment commitment cannot
   declare those labels.
7. `R in [2.15,2.5]`, `F <= 0.8`, and read amplification `<2` are applied to
   every claimed SwiGLU-MoE architecture family, not only the Qwen subset.
8. Each family is compared with its strongest complete control panel using
   `s=-0.5 log2(F)`.  Acceptance requires at least `0.03 bpw` of model-minus-
   strongest-control advantage, and Qwen must independently satisfy `F<=0.8`.
9. The source gate invokes the installed current hook at both `N=2**20` and
   `N=2**21`.  A separate parity worker compares the frozen v1 order against a
   CPU Gosper fixed-weight enumeration and an independently implemented CuPy
   byte-LUT popcount order over every phase.

## Authority topology

The physical validator needs four independent capabilities:

- an out-of-band SHA-256 for this exact source manifest;
- an out-of-band SHA-256 for the canonical experiment commitment;
- an out-of-band scientific-provenance capability SHA-256 owned by the
  independent auditor;
- out-of-band manifest/root/receipt hashes for a successfully executed
  independent audit of the exact production decoder and this exact
  instrumentation launcher.

The commitment owns only literal packet identity and a reference to a
scientific capability ID.  It does not own source paths, tensor hashes,
checkpoint hashes, model kind, architecture family, control kind, generator,
seed, moment record, rate, MSE, F, or read counts.

All source-specific state must be inside the routed expert packet.  A decoder
audit must certify that the decoder is fixed and universal and contains no
Qwen-specific tables.  The launcher snapshots the packet, request, decoder,
and itself into a fresh directory; the decoder sees only those paths and its
output directory.

## Source gate

After this package's manifest SHA-256 has been pinned outside the package:

```bash
/workspace/int2-cupy-venv/bin/python -I -B run_source_gate.py \
  --package /workspace/strata_rm_global_swap_v2_authority \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --v1-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v1_hardened \
  --review-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v1_hardened_reproducibility_review_20260902 \
  --external-root /workspace/INT2__compression \
  --run-current-integration --run-real-cupy \
  --output /tmp/strata_rm_global_swap_v2_source_gate.json
```

The source-only hostile suite is:

```bash
python -I -B test_source_only.py
```

Neither command accepts a model, checkpoint, packet, tensor, or payload path.

## Physical gate (held)

`authority_v2.validate_physical_bundle` is the only physical result entry
point.  It requires `AUDIT_LITERAL_GLOBAL_RM_SWAP_RESULT_V2` plus every
out-of-band capability described above.  No valid production decoder-audit or
scientific-provenance capability is bundled here, so this frozen package
cannot emit a Qwen or universal result on its own.

The current honest disposition is:

```text
FROZEN_V2_AUTHORITY_SOURCE_ONLY__NINE_REVIEW_GAPS_REPAIRED_IN_SOURCE__AWAIT_EXECUTION__HOLD_PAYLOAD_AND_RD
```
