# Nested composite super-oracle

## Decision

**No charged survivor.**  A source-locked composite of the strongest distinct
structures tested so far does not reach the required

```text
F = MSE * 2^(2R) <= 0.8
```

on the pinned 18-matrix Qwen3-30B-A3B panel.  The best honest result is the
joint role-innovation plus whole-matrix polar model:

| Quantity | Best charged composite | Required |
|---|---:|---:|
| `F` | **0.9363976210** | **<= 0.8** |
| Rate-equivalent `s=-0.5 log2(F)` | **0.0474034129 bpw** | **>= 0.1609640474 bpw** |
| Fraction of required `s` | **29.45%** | **100%** |
| MSE at 2.5 bpw | **0.0292624257** | **<= 0.025** |
| Excess over target MSE | **17.05%** | **<= 0%** |
| Cold compressed expert read | **1.00278x** | **<2x** |

Because every selected component remains active over 2.15--2.5 bpw, its
normalised `F` is constant on that interval:

| Physical rate | Ideal charged MSE | Target MSE | `F` |
|---:|---:|---:|---:|
| 2.15 | 0.0475369507 | 0.0406126198 | 0.9363976210 |
| 2.30 | 0.0386120021 | 0.0329876978 | 0.9363976210 |
| 2.50 | 0.0292624257 | 0.0250000000 | 0.9363976210 |

No finite quantizer or GPU integration is justified by this result.  It is a
negative architecture gate, not a converse for every possible codec.

## Why this is a real composite rather than added gains

The construction never adds `s` values or multiplies separately measured
`F`s.  Every candidate is reconstructed from the authenticated BF16 source,
decomposed once into nested components, and passed to one ideal Gaussian
reverse-waterfiller.

For each expert the three logical matrices are represented as

```text
G = gate_proj, U = up_proj, D = down_proj.T
```

and the modules are applied in this order:

1. **Joint role innovation.**  The exact source-specific 3x3 KLT of aligned
   `(G,U,D)` triples is applied at every coefficient.  It is orthogonal in the
   requested source Frobenius metric and exposes the strongest lossless linear
   role predictor without circularly revealing two exact roles to the third.
2. **Expert-local STRATA.**  Natural 2,048-value rows are stably ranked by
   energy and split into eight equipopulous 96-row units per channel.  Units
   have disjoint row support, so their measured energies add exactly.
3. **Polar/Stiefel split.**  Inside the current matrix or STRATA unit,
   `A=H Q` is modelled by `H_hat=cI+A_k`.  For every rank, all contiguous
   unmodelled singular-value windows are exhausted.  The exact mean `c` makes
   `H_hat Q` orthogonal to `(H-H_hat)Q`.  Manifold and normal dimensions are
   charged explicitly.
4. **One reverse-waterfill.**  All final component energies and dimensions are
   allocated jointly after deducting the literal header, label, role-angle,
   rank, and window-index bits from the requested physical rate.

The exact SwiGLU Up/Down gauge was also audited.  If
`Uc=aU, Dc=D/a`, the inverse source metric is
`diag(1,a^-2,a^2)`.  Its square root maps the canonical coordinates exactly
back to `(G,U,D)`.  Therefore the gauge contributes no additional source-MSE
advantage beyond the source-space role basis.  Counting both would be double
counting, so the result records its scales but assigns it zero separate gain.

The 2-D scale-field branch is deliberately absent because it was owned by the
concurrent `spectral_scale_field` experiment.  This package does not duplicate
or contaminate that experiment.

## Charged results

The following table is at 2.5 bpw and includes each variant's explicit side
ledger.  `F` is the physical-rate score, not a payload-only score.

| Modules | `F` | `s` (bpw) | Ideal MSE | Explicit side (bpw) | Decision |
|---|---:|---:|---:|---:|---|
| Energy-only baseline | 0.9962155903 | 0.0027350531 | 0.0311317372 | 0.0011574074 | Kill |
| Role/gauge | 0.9795912916 | 0.0148740731 | 0.0306122279 | 0.0011675799 | Kill |
| STRATA | 0.9917538692 | 0.0059729870 | 0.0309923084 | 0.0026222512 | Kill |
| Role/gauge + STRATA | 0.9685274411 | 0.0230675849 | 0.0302664825 | 0.0026324237 | Kill |
| Whole-matrix polar | 0.9520339564 | 0.0354575317 | 0.0297510611 | 0.0011701231 | Kill |
| **Role/gauge + whole-matrix polar** | **0.9363976210** | **0.0474034129** | **0.0292624257** | **0.0011802956** | **Best; kill** |
| STRATA-local polar | 0.9882463659 | 0.0085286750 | 0.0308826989 | 0.0026934588 | Kill |
| Role/gauge + STRATA-local polar | 0.9649791066 | 0.0257151944 | 0.0301555971 | 0.0027036314 | Kill |

This table exposes a useful non-additivity result.  Whole-matrix polar has a
9.687% normal codimension carrying 3.180% of source energy.  Splitting first
into 96-row STRATA units leaves only a 1.054% selected normal codimension
carrying 0.882% of energy.  The smaller polar problems are much closer to
isotropic per degree, so STRATA-local polar is worse—not the product of the
individual reported gains.

## Deliberately source-leaky envelope

The requested super-oracle also asks whether an intentionally favourable
union can reach the target.  It can, but only by omitting a large
source-specific component from the physical stream.

For the minimal one-module whole-matrix polar construction, reveal the exact
normal correction for free and spend all 2.5 bpw on the manifold:

| Quantity | Free-normal envelope |
|---|---:|
| `F` | **0.6639684519** |
| Ideal MSE | 0.0207490141 |
| Manifold dimension / energy | 90.3126% / 96.8199% |
| Free normal side DOF / energy | 9.6874% / 3.1801% |
| FP16 storage for omitted side | **1.5499894884 bpw** |
| Total if added to the 2.5-bpw payload | **4.0499894884 bpw** |

This is not an operational pass.  With the actual 18-component all-active
waterfill, the normal side may consume at most **0.1214186091 bpw** before
framing, or **0.1202484861 bpw** after the charged header/rank ledger, while
retaining `F<=0.8` under a strict total cap of 2.5 bpw.  The naive FP16
representation is therefore **12.89x too large**.  Compressing both manifold
and normal optimally instead of giving either away is precisely the charged
polar waterfill above, `F=0.9520339564`.

Adding role innovation improves the free-normal envelope to `F=0.6536033150`.
The 9.6822% normal side is 1.5491491247 bpw in FP16 and must fit within
0.1304961734 bpw after explicit framing.  This does not create an operational
survivor.

The other free direction—revealing the full manifold and encoding only the
normal—is vastly more source-leaky: it omits roughly 90.3% of source DOF and
would require about 14.45 bpw at FP16.  It is retained in `result.json` only as
a sanity-check ceiling.

## Side rate and MoE read feasibility

All real module state is expert-local.  No candidate requires another expert's
payload.  At exactly 2.5 bpw, equal expert frames plus a pessimistically cold
4 KiB global manifest give read amplification `1.0027777778x`; all candidates
are comfortably below 2x.  The explicit side bytes are already inside the
physical rate.

This is compressed-object accounting.  Role and polar reconstruction are
dense.  An inference implementation would have to tile or fuse them into the
expert GEMMs; materialising BF16 weights or intermediate transforms in HBM
would invalidate a total-traffic interpretation of the read figure.

The leaky FP16 side is different: it is omitted from the scored container.  If
stored, it raises both rate and reads and must be charged before any locality
claim.

## Reproduction

The experiment was CPU-only on the provided RunPod while the sealed CuPy GPU
encoder ran independently:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/composite_superoracle/composite_superoracle.py \
  --root /workspace/INT2__compression \
  --output INT2_Q_C/research/composite_superoracle/result.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/composite_superoracle/verify_result.py \
  --root /workspace/INT2__compression \
  --result INT2_Q_C/research/composite_superoracle/result.json \
  --output INT2_Q_C/research/composite_superoracle/verification_receipt.json
```

The independent verifier passed.  It does not import the experiment: it
rehashes all 18 BF16 files, rebuilds all four source geometries, recomputes all
324 singular spectra and complete rank curves, repeats every selected joint
waterfill, checks every free-component envelope, and reselects the aggregate
decision.

Artifacts:

- `composite_superoracle.py` — source-locked CPU experiment;
- `result.json` — sealed complete spectra, rank curves, selections, component
  allocations, side ledgers, and decisions;
- `verify_result.py` — independent source/geometry/arithmetic verifier;
- `verification_receipt.json` — passing verification receipt;
- `ARTIFACT_HASHES.json` — artifact identities.

## Claim boundary

The charged scores are ideal-RD architecture oracles, not achieved finite-code
MSE.  They favour the candidate with continuous manifold coordinates, exact
charts, infinite precision, adaptive source-specific ranks, and asymptotic
Gaussian coding.  Failure under those assumptions early-stops this composite.

The free-component scores are even more permissive and explicitly decoder
illegal unless their omitted source-specific side is serialized.  They identify
the only potentially interesting bottleneck—procedurally predicting the polar
normal field in at most about 0.120 bpw—but do not establish that such a
predictor exists.
