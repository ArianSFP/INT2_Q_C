# Polar normal-field predictor oracle

## Decision

**Hard kill.**  The only source-leaky pass exposed by the nested composite
oracle was an exact, free polar normal correction.  This follow-up tests
whether that field can be predicted or represented from held-out shared,
analytic, or decoder-visible structure inside the available side budget.

It cannot.  At 2.5 bpw:

| Gate | Best candidate | Field rate | `F` | Required |
|---|---|---:|---:|---:|
| Free exact coefficients, FP16-sized field `<=0.117869` bpw | Identity band 12 | 0.100769 bpw | **0.9335045648** | `<=0.8` |
| Impossible exact one-bit continuous coefficients | Identity band 128 | 0.057739 bpw | **0.8621028948** | `<=0.8` |
| Exact coefficients charged at FP16, field inside gate | Held-out radial implicit | 0.000010 bpw | **0.9520471384** | `<=0.8` |

The one-bit line is the strongest rejection.  It grants every arbitrary
continuous source-specific coefficient exact reconstruction for one bit, an
impossible finite representation, yet still misses the target by 7.76% in
`F`.  No finite coefficient quantizer, implicit model, or GPU integration is
warranted for these families.

The conservative prerequested field gate is `0.1178690728 bpw`.  The corrected
18-component allowance after the existing polar header/rank ledger is slightly
larger, `0.1202484861 bpw`; no decision changes at that boundary.

## What was predicted

For each authenticated canonical matrix, the selected polar decomposition is

```text
W = H Q
H_hat = c I + A_k
W = H_hat Q + N Q
```

where `N=H-H_hat` is the polar normal field.  The decoder already pays for the
manifold `H_hat Q`.  The composite free-normal envelope reached `F=0.66397`
only by revealing all of `N Q` without rate.

This experiment reconstructs the exact source-specific symmetric `N`, applies
an orthogonal predictor `N_hat`, and jointly reverse-waterfills the manifold
and unpredicted `(N-N_hat)Q`.  Predictor coefficient DOF is removed from the
coded normal dimension—a favourable grant.  Explicit polar framing and every
private coefficient ledger are deducted from the 2.5-bpw physical budget.

## Held-out protocol and controls

The held-out unit is a complete expert triplet.  When a predictor uses shared
normal matrices, eigenbases, or coordinate statistics, none of the target
expert's Gate, Up, or Down normal fields enters the fit.  Same-layer router
weights are allowed because the router is decoder-visible before an expert is
fetched.

Every candidate is also run on deterministic independent Gaussian matrices
matching each target's exact mean and centered energy.  Controls reuse the
target's polar rank and independently select their best singular window.  This
separates Qwen structure from finite-aspect polar geometry.

The tested families are:

- identity and analytic DCT diagonal/banded symmetric fields;
- held-out global and role-specific energy eigenbases;
- held-out global and role normal-template spans;
- procedural 2-D low-frequency DCT fields;
- a tiny coordinate-conditioned model `N_ij=f_role(|i-j|)` with 2,304 shared
  values and one target scale;
- same-layer router PCA right subspaces; and
- the exact routed-expert router row.

All source-specific coefficient values are granted lossless even when their
ledger says one bit or FP16.  Shared tables use favourable full-model
amortisation across 48 layers and 128 experts.

## Main evidence

### Budget-fitting candidates

| Candidate | Normal energy captured | Gaussian control | Free `F` | FP16 field | Charged FP16 `F` |
|---|---:|---:|---:|---:|---:|
| Identity band 12 | **4.64369%** | 3.35955% | **0.9335046** | 0.100769 | 1.0744098 |
| DCT low-frequency triangle 192 | 3.14969% | 3.15318% | 0.9356779 | 0.094727 | 1.0678176 |
| Router PCA rank 8 | 0.30418% | 0.39348% | 0.9428160 | 0.062500 | 1.0284975 |
| Target router row rank 1 | 0.04662% | approximately Gaussian | 0.950879 | 0.007812 | >0.95 |
| Held-out role normal span | 0.00149% | 0.00207% | 0.9520254 | 0.000051 | 0.9585591 |
| Tiny radial implicit | 0.00051% | 0.00020% | 0.9520320 | 0.000010 | **0.9520471** |

The small identity-band excess over Gaussian is real: Qwen captures 1.284
percentage points more normal energy at width 12.  It is nowhere near the
amount required, and spending its FP16 side rate makes distortion worse than
the original charged polar oracle (`F=0.9520339564`).

Router conditioning is actively unhelpful on the normal field: rank 8 captures
less Qwen energy than its matched Gaussian control.  The 2-D DCT result is
indistinguishable from Gaussian.  Held-out templates and the tiny coordinate
model are effectively zero.

### Why the free passes do not survive rate

Two high-capacity projections cross `F=0.8` only while their coefficients are
free:

| Candidate | Free `F` | FP16 field | One-bit field | `F` after impossible one-bit charge |
|---|---:|---:|---:|---:|
| Identity band 128 | 0.7918929 | **0.923828 bpw** | 0.057739 bpw | **0.8621029** |
| Router PCA rank 128 | 0.7886328 | **1.000000 bpw** | 0.062500 bpw | **0.8649922** |

Thus the apparent free-predictor survivors need roughly eight times the FP16
field allowance.  Even the impossible one-bit exact ledger fits the nominal
field budget but consumes enough payload rate to lose the target.  This is not
a quantizer-engineering gap.

### Source-adaptive union

For extra optimism, each matrix may choose the best budget-fitting family with
free mode labels.  Qwen selects identity band 12 for all 18 matrices:

```text
captured normal energy = 4.6436887%
free-coefficient F      = 0.9335045648
charged FP16 F          = 1.0744098327
```

The matched-Gaussian adaptive union captures 3.40456% and has free
`F=0.9332183`.  Adaptive selection does not reveal a hidden composite win.

## Rate and MoE read ledger

Identity/DCT and router bases are analytic or already decoder-visible, so they
add no shared cold read.  Their private coefficient fields are expert-local.
An expert-frame layout remains near the composite checkpoint's approximately
`1.003x` cold compressed-object read when the field stays inside the physical
rate.

The tiny radial table is only 4,608 bytes.  Dense held-out role bases or normal
template spans can exceed 2x on a cold uncached read; their cached/full-model
amortised ledgers are reported separately in `result.json`.  They already fail
the MSE gate, so caching cannot rescue the architecture.

As elsewhere, compressed-object locality does not include materialising dense
normal corrections.  A surviving design would require reconstruction fused
with expert GEMM consumption.  No design survived here.

## Reproduction and verification

CPU-only RunPod experiment:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/polar_normal_predictor/polar_normal_predictor.py \
  --root /workspace/INT2__compression \
  --base-result INT2_Q_C/research/composite_superoracle/result.json \
  --router-dir qwen_aux_context_tensors/router_blocks \
  --output INT2_Q_C/research/polar_normal_predictor/result.json
```

Verification:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/polar_normal_predictor/verify_result.py \
  --root /workspace/INT2__compression \
  --result INT2_Q_C/research/polar_normal_predictor/result.json \
  --base-result INT2_Q_C/research/composite_superoracle/result.json \
  --router-dir qwen_aux_context_tensors/router_blocks \
  --output INT2_Q_C/research/polar_normal_predictor/verification_receipt.json
```

The verifier passed.  It rehashed all 18 sources, six router matrices, and the
sealed parent result; rebuilt 18 Qwen and 18 matched-Gaussian polar normals;
recomputed all 64 coefficient/rate/waterfill ledgers; and independently rebuilt
the decisive identity-band, router, routed-row, and radial projections.

Artifacts:

- `polar_normal_predictor.py` — source-locked CPU experiment;
- `result.json` — sealed candidate/control/ledger result;
- `verify_result.py` — source, projection, and arithmetic verifier;
- `verification_receipt.json` — passing verification receipt;
- `ARTIFACT_HASHES.json` — frozen artifact identities.

## Claim boundary

This is an ideal-RD early-kill oracle, not emitted compressed weights or
achieved finite-code MSE.  It is deliberately more favourable than a codec:
continuous target coefficients are exact, one-bit/FP16 storage incurs no
coefficient quantisation error, predictor subspace DOF disappears from the
residual, and Gaussian coding is asymptotic.  Failure under these assumptions
closes the listed polar-normal predictor families, not every conceivable
procedural generator.
