# Direct-output codebook whole-expert stage 0

## Status and scope

This is a source-only, not-yet-executed CuPy gate. It is deliberately separate
from the killed latent4 nonlinear meta-decoder cell. No decoder or encoder can
hide representational failure here: the stored object is the literal
`32768 x 8` FP16 reconstruction table searched by exact nearest neighbor.

A negative result rejects only this fixed initialization, minibatch/Lloyd, and
empty-cluster recipe. It is not a converse for every K-means implementation,
arbitrary VQ, SoftBinary Coding, or nonlinear codec. A positive result remains
an ideal-residual feasibility result on two held-out experts, not model-wide
generalization or a finite codec.

The runner requires a literal authorization argument and must not be launched
until GPU coordination.

## Bound panel and split

The only accepted plan is
`blind_protocol_v2/unblinded/source_hashes.lock.json` beneath `--root`, with
46,013 bytes and SHA-256
`bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23`.
Payload paths remain relative to the lock parent. Down is transposed after load
so every logical matrix is `768 x 2048`.

The whole-expert split was frozen before results:

| Slot | Layer/expert | Split |
|---:|---:|---|
| 0 | 5 / 18 | fit |
| 1 | 12 / 7 | holdout |
| 2 | 18 / 20 | fit |
| 3 | 28 / 83 | fit |
| 4 | 36 / 76 | holdout |
| 5 | 45 / 41 | fit |

Every natural 2,048-value row has a charged FP16 mean/RMS pair. Source and
Gaussian matrices use identical row-RLN decoding. The Gaussian control is
matched row by row before any FP16 moment rounding and is trained independently
with the identical fixed recipe.

## Frozen direct K-means recipe

- Contiguous row-major vectors of dimension eight.
- `K=32768`, hence 15 index bits/vector and exactly 1,105,920 index
  bytes/expert.
- Two fixed seeds: `2026090111`, `2026090112`.
- Initialization takes an almost-equal no-replacement modular-stride sample
  from each of the 12 fit matrices; it never reads a holdout identity.
- 1,024 deterministic round-robin minibatches of 4,096 vectors.
- Exact tiled assignment in every update; code tiles contain 2,048 rows.
- Deterministic sorted segment sums; no atomic scatter accumulation.
- At each checkpoint, clusters empty since initialization or unused in two
  consecutive windows are reseeded from a fixed reservoir of the largest
  observed assignment errors.
- One complete exact Lloyd assignment/update pass over all four fit experts.
- Final table is rounded to canonical little-endian FP16 before scoring.
- Every vector in both held-out experts is then searched exactly against that
  finite table. The minibatch encoder is not used for scoring.

The fixed fit-only collapse checkpoints are:

| Checkpoint | Maximum fit-probe q | Minimum probe codes used |
|---:|---:|---:|
| 128 | 0.30 | 2,048 |
| 256 | 0.25 | 4,096 |
| 512 | 0.20 | 8,192 |
| 1,024 | 0.16 | 12,288 |
| Full Lloyd pass 1 | 0.14 | 16,000 |

A checkpoint failure is reported but exact held-out scoring is still run. It
does not silently substitute for the target decision.

With a 4,096-vector evaluation batch and 2,048-code tile, distance scratch is
about 32 MiB. Minibatch training uses the same tiling. Expected peak device
memory is below 2 GiB; the full exact scans make runtime approximately tens of
minutes on an RTX 5090.

## Physical rate and reads

The global object is exactly 528,384 bytes: a 4,096-byte canonical header plus
the 524,288-byte FP16 table. Each expert-local frame additionally contains
9,216 row-moment bytes, a 64-byte header, and 1,105,920 index bytes.

For the six-expert evidence panel the fixed prefix is exactly
`2.0400390625 bpw`; therefore the favorable first-stage requirement is
`q <= 0.047300320854109984`.

| R | Physical bytes | Max local frame | Residual bytes | Residual bpw | Cold bytes | Cold amplification |
|---:|---:|---:|---:|---:|---:|---:|
| 2.15 | 7,608,730 | 1,180,058 | 389,146 | 0.10996105 | 1,712,128 | 1.35012913x |
| 2.30 | 8,139,572 | 1,268,532 | 919,988 | 0.25996116 | 1,798,144 | 1.32548296x |
| 2.50 | 8,847,360 | 1,386,496 | 1,627,776 | 0.45996094 | 1,916,928 | 1.30000000x |

The same byte format projected arithmetically over a hypothetical 128-expert
layer has prefix `1.8977322048611112 bpw` and requires
`q <= 0.05761576759174624`. This is not a generalization claim.

| R | Whole-layer bytes | Max local frame | Residual bytes | Residual bpw | Cold bytes | Cold amplification |
|---:|---:|---:|---:|---:|---:|---:|
| 2.15 | 162,319,565 | 1,263,994 | 19,045,581 | 0.25226780 | 1,794,048 | 1.41472868x |
| 2.30 | 173,644,186 | 1,352,468 | 30,370,202 | 0.40226780 | 1,884,160 | 1.38888889x |
| 2.50 | 188,743,680 | 1,470,432 | 45,469,696 | 0.60226780 | 1,998,848 | 1.35555556x |

Cold reads include the whole global table plus the largest local expert frame
rounded to 4 KiB, with no cache assumption.

## Favorable oracle and decision

Let `q` be pooled held-out residual energy after exact search of the canonical
FP16 table. Every remaining physical bit is granted to an ideal continuous
Gaussian residual code. Thus

```text
F_oracle = q * 2**(2 * fixed_prefix_bpw)
s_oracle = -0.5 * log2(F_oracle)
```

- **KILL:** the better of the two fixed seeds still has `F_oracle > 0.8`.
- **PROMOTE_TO_FRESH_AUXILIARY_CONFIRMATION_ONLY:** neither source run trips a
  collapse checkpoint, both have `s >= 0.18096404744368115`, every held-out
  expert has `F <= 0.8`, and both source-minus-Gaussian `s` values are positive.
- Otherwise: **HOLD_INCONCLUSIVE**.

## Source-only verification

```bash
/usr/bin/python3 -B -I \
  research/direct_output_codebook_whole_expert_stage0_v0/verify_source.py \
  --root /workspace/INT2__compression
```

## Deferred launch

Run only after coordination:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
/workspace/int2-cupy-venv/bin/python -B -I \
  research/direct_output_codebook_whole_expert_stage0_v0/direct_output_codebook_stage0.py \
  --root /workspace/INT2__compression \
  --output /workspace/direct_output_codebook_whole_expert_stage0_v0_run \
  --authorization OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_DIRECT_OUTPUT_CODEBOOK_STAGE0_V0
```
