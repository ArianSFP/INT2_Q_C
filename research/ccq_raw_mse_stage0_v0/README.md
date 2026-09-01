# CCQ raw-MSE stage-0 gate

Status: **SOURCE-ONLY, SEALED FOR COORDINATED RUNPOD PREFLIGHT; NOT EXECUTED.**

This package freezes one bounded CuPy test of the only CCQ branch that survives
the exact rate ledger: `(L=6,N=4,S=3)` Code Cluster. The primary-source and
containment reasoning is in `RESEARCH_FINDING.md`.

No model payload was opened while researching or sealing this package. The
runner requires a literal authorization, validates the authenticated 18-matrix
lock before importing CuPy, refuses a nonempty output, and resolves payloads
only beneath the lock parent.

## Frozen evidence split and control

The whole-expert split is fixed before results:

| Slot | Layer/expert | Use |
|---:|---:|---|
| 0 | 5/18 | fit diagnostic |
| 1 | 12/7 | holdout decision |
| 2 | 18/20 | fit diagnostic |
| 3 | 28/83 | fit diagnostic |
| 4 | 36/76 | holdout decision |
| 5 | 45/41 | fit diagnostic |

All source-adaptive CCQ fields are expert-local transmitted fields; the fit
split does not leak them into holdout identities. Fit experts only report the
frozen recipe's diagnostic behavior. The hard decision uses the two complete
held-out experts.

Each matched Gaussian matrix independently matches every natural 2,048-value
row's mean and centered RMS before applying the same role orientation, encoder,
canonical field casts, packet round-trip, and score. Gaussian controls are
empirical algorithm controls, not a lower bound or Qwen evidence.

## Frozen paper-derived encoder

The official repository exposes the field shapes and dequantizer but no
offline encoder. This gate therefore freezes the following deterministic cell:

1. Orient Gate/Up as `2048 x 768` and Down as `768 x 2048`, matching the
   released `K x N` kernels.
2. Initialize one positive scale per 64-weight K-axis group.
3. Run three coordinate passes. For fixed scale, an exact 64-state dynamic
   program finds the minimum-MSE 15-bit overlapping-window word for every
   four-vector. Then update each continuous group scale by least squares.
4. Quantize group scales to uint4 times a canonical FP16 per-channel
   super-scale.
5. Uniformly min/max-map the resulting 15-bit words per output channel into a
   byte, storing canonical FP32 code scale and zero point.
6. Run two exact byte-assignment/group-scale refinement passes, serialize the
   released fields, parse them back, and score only that reconstruction.

This is favorable in one explicit respect: scoring multiplies the canonical
fields in FP32 and does not charge deployment BF16 multiply rounding. A kill is
therefore safe for this frozen cell; a survivor still needs an official-format
finite residual implementation and independent decoder audit.

## Rate, ideal residual, and reads

The six-expert fixed prefix is 7,518,592 bytes,
`2.1245298032407407 bpw`. Its pooled first-stage requirement is
`q <= 0.0420722358191473`.

| Final R | Physical bytes | Ideal-residual bytes | Max local frame | 4-KiB cold bytes | Cold amplification |
|---:|---:|---:|---:|---:|---:|
| 2.15 | 7,608,730 | 90,138 | 1,267,439 | 1,273,856 | 1.0045219110x |
| 2.30 | 8,139,572 | 620,980 | 1,355,913 | 1,363,968 | 1.0054346838x |
| 2.50 | 8,847,360 | 1,328,768 | 1,473,878 | 1,478,656 | 1.0027777778x |

Every byte, unequal division remainder, padding page, fixed field, and ideal
residual allocation is charged. The residual is a containing information
oracle, not emitted codec evidence. Since

```text
F_oracle = q * 2**(2 * fixed_prefix_bpw),
```

the same hard-kill identity applies at every final rate in `[2.15,2.5]`.

During held-out scoring, accumulated SSE divided by the complete held-out
source energy is monotone. The runner records the first matrix where that
lower bound alone crosses `F=0.8`; no unscored matrix can recover it. The full
frozen control and packet audit still complete so the result remains
diagnosable.

- `KILL`: pooled held-out `F_oracle > 0.8`.
- `PROMOTE_TO_FRESH_AUXILIARY_CONFIRMATION_ONLY`: pooled and both expert
  `F<=0.8`, `s>=0.18096404744368115`, and source-minus-Gaussian `s>0`.
- Otherwise `HOLD_INCONCLUSIVE`.

## Source-only preflight

From `/workspace/INT2__compression`:

```bash
/usr/bin/python3 -B -I \
  research/ccq_raw_mse_stage0_v0/verify_source.py \
  --root /workspace/INT2__compression

/usr/bin/python3 -B -I \
  research/ccq_raw_mse_stage0_v0/test_source_only.py
```

The verifier is standard-library only. It checks the exact non-link closure,
manifest, receipt, primary-source bindings, authenticated-plan identity,
hybrid rate kill, Code-Cluster byte/rate/read arithmetic, split, source-path
semantics, and static-compiles the runner without importing NumPy or CuPy.

## Deferred coordinated launch

Do not run until coordinated:

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NVIDIA_TF32_OVERRIDE=0 \
    CUPY_ACCELERATORS= \
  /workspace/int2-cupy-venv/bin/python -B -I \
  research/ccq_raw_mse_stage0_v0/ccq_stage0.py \
  --root /workspace/INT2__compression \
  --output /workspace/ccq_raw_mse_stage0_v0_run \
  --authorization OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_CCQ_RAW_MSE_STAGE0_V0
```

Expected peak device memory is below 2 GiB. The exact 64-state dynamic program
and 256-entry channel-table searches should take roughly 10--30 minutes on an
RTX 5090; this is an estimate until the first coordinated run. Output is
written only beneath the supplied absent/empty directory: `source_prefix.bin`,
`gaussian_prefix.bin`, and `result.json`.
