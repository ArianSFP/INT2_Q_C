# Reproducibility review: hardened global STRATA RM swap v1

Date: 2026-09-02

This is an independently authored, source-only review of producer root
`980a5f1d272ca5ffc7b4d35e7c234a86994d135fcacaf0d47a8b3e00fc3d4f14`
and manifest SHA-256
`4c2c5b371b1b9661d371de607e6a650f8c43fe0128726476854c2eb2ca560c85`.

It does not modify the producer and has no model/checkpoint discovery code.
Its synthetic fixture contains twelve tiny BF16 values and is explicitly not
Qwen, a Gaussian control, a rate-distortion result, or payload evidence.

## Review coverage

- exact regular-file manifest closure and dependency pins;
- strict/canonical JSON serialization;
- worker source snapshotting and isolated interpreter/environment controls;
- current-hook source identity checks;
- synthetic literal packet decode and canonical byte re-encode;
- exact-BF16-source/FP64-reconstruction numerical recomputation;
- real-CuPy origin, runtime, device, synchronization, and full-order checks;
- the stated source-only scope and every remaining production hold.

`REVIEW_ASSESSMENT.md` separates verified mechanisms from correctness and
authority gaps.  In particular, a source-only pass cannot be promoted to a
Qwen result.

## Source-only execution

From a Python environment containing the repository's test dependencies:

```bash
python -I -B run_review.py \
  --producer /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v1_hardened \
  --external-root /workspace/INT2__compression \
  --output /tmp/strata_rm_global_swap_v1_reproducibility_review.json
```

The suite launches only the producer's tiny synthetic fixture decoder.  It
must not be pointed at model or packet payload directories.

## Optional trusted-runner CuPy execution

The accelerator review is a separate no-payload command:

```bash
/workspace/int2-cupy-venv/bin/python -I -B run_real_cupy_review.py \
  --producer /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v1_hardened \
  --external-root /workspace/INT2__compression \
  --output /tmp/strata_rm_global_swap_v1_real_cupy_review.json
```

It independently rebuilds the complete `2**20` and `2**21` row orders on the
CPU and compares their byte hashes with the isolated producer worker.  This is
trusted-runner provenance, not cryptographic hardware attestation.

## Current disposition

The review source is frozen but its Python/CuPy suites are unexecuted.  The
only admissible disposition before receipts exist is:

```text
FROZEN_REPRODUCIBILITY_REVIEW_UNEXECUTED__CORRECTNESS_GAPS_RECORDED__HOLD_PAYLOAD_AND_PHYSICAL_RESULT
```
