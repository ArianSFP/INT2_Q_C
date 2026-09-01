# Exact-gauge / joint Gate–Up–Down polar oracle

## Decision

**Hard kill.** Quotienting the exact SwiGLU neuron symmetries and fitting a
single coupled Gate–Up–Down polar manifold does not approach the rate-relative
target on the pinned six-expert Qwen panel.

The strongest result is deliberately more favourable than an implementable
codec: it gives the gauge and manifold charts losslessly, uses continuous
coordinates, selects source-specific ranks, and applies ideal asymptotic
Gaussian reverse-waterfilling. Even so, the adaptive oracle reaches only

| Physical rate | Ideal relative MSE | `F=MSE*2^(2R)` | `s=-0.5log2(F)` | Required MSE |
|---:|---:|---:|---:|---:|
| 2.15 | 0.0501265413 | 0.9874081807 | 0.0091407485 | 0.0406126198 |
| 2.25 | 0.0436376888 | 0.9874081807 | 0.0091407485 | 0.0353553391 |
| 2.50 | 0.0308565056 | 0.9874081807 | 0.0091407485 | 0.0250000000 |

Success requires `F<=0.8`, or `s>=0.1609640474 bpw`. The free-side oracle
supplies only **5.68%** of that rate-equivalent advantage and misses by
`0.1518232990 bpw`. No finite-code or GPU experiment is justified.

## Distinct hypothesis tested

For an expert, put all roles in the common neuron orientation:

```text
G = gate_proj             in R^(768 x 2048)
U = up_proj               in R^(768 x 2048)
D = down_proj.T           in R^(768 x 2048)
```

The SwiGLU expert is exactly invariant to a shared neuron permutation and to
the continuous diagonal action

```text
U -> A U,       D -> A^-1 D,
```

because the up activation is linear and its scale cancels the corresponding
down column. The probe chooses the unique positive equal-norm gauge

```text
a_j = sqrt(||d_j||_2 / ||u_j||_2),
||a_j u_j||_2 = ||d_j/a_j||_2,
```

then forms one joint matrix

```text
X = [G, A U, A^-1 D] in R^(768 x 6144).
```

It computes the exact polar decomposition `X=H Q`, `Q Q.T=I`, and models the
entire triplet by

```text
H_hat = c I + A_k,       rank(A_k)=k.
```

This differs from the earlier single-matrix Stiefel/Gram screen: one row frame
now couples all three semantic roles *after* quotienting an exact architectural
symmetry. It also differs from cross-expert neuron prediction: no other
expert's weights are used as a reference.

The equal-norm cross-section removes 768 continuous coordinates, but source
reconstruction requires retaining the 768 gauge coordinates, so the net
dimension saving is exactly zero. A shared permutation likewise has to be
inverted for source-domain scoring; its exact enumerative information is
`ceil(log2(768!)) = 6,260 bits` per expert.

## Exact source-metric rank search

The canonical gauge is not an orthogonal transform, so scoring Frobenius error
in canonical coordinates would be invalid. The probe instead derives the
inverse-gauge quadratic form. If `s_i` are joint polar singular values and an
unmodelled set shares singular value `c`, its original-source residual is

```text
(1-c/s)^T K (1-c/s),
```

where `K` includes Gate with unit metric, Up with `A^-2`, and Down with `A^2`.
For every rank `k=0,...,766`, all contiguous unmodelled eigenspectrum windows
are exhausted. Three two-dimensional prefix sums solve the best source-metric
`c` and residual for every window exactly. Ranks are then selected by repeated
exact coordinate minimization of the panel reverse-waterfill objective.

At 2.5 bpw, the best common rank is 290 (`F=0.98745719`). Per-expert adaptive
ranks `[268, 282, 315, 323, 297, 251]` improve this only to `F=0.98740818`.
All twelve manifold/normal components remain active, so the same normalized
`F` holds throughout the tested rate interval.

The measured gauges are already close to balanced: per-expert log-gauge
standard deviations are only `0.01212–0.01676`, and all 4,608 scales lie in
`[0.82633, 1.10313]`. There is no large scaling orbit to exploit.

## Held-out and matched Gaussian controls

The same gauge, polar, rank-window, source-metric, and waterfill procedure was
run on deterministic independent Gaussian triplets. Finite controls were
rescaled to match their requested first two moments exactly.

| Control | Moment protocol | `F` at 2.5 | `s` at 2.5 |
|---|---|---:|---:|
| Target-matched | Each role matches that target expert role's exact mean and variance | 0.9817222112 | 0.0133066193 |
| Held-out | Each target role uses moments fitted only from the other five experts | 0.9845774883 | 0.0112116706 |

Both Gaussian controls beat the Qwen source oracle. Averaging the two controls
and all three rates gives `s=0.0122591449`; Qwen minus control is
`-0.0031183965 bpw`. Thus even the small raw gain is a generic finite-aspect
polar effect, not favourable source structure.

## Exact side and cold-read ledger

The survival test gives side information away. Separately, the result records
a byte-exact expert-local layout to test the locality constraint. The global
manifest is adjusted by at most five bytes so six equal frames fill the exact
container budget. Every frame contains the following explicit side objects:

| Per-expert object | Bytes |
|---|---:|
| Frame header | 64 |
| 768 FP32 gauge values | 3,072 |
| Enumerative shared permutation (`6,260` information bits) | 783 |
| Rank label (`uint16`) | 2 |
| Payload directory | 32 |
| CRC32 | 4 |
| **Total** | **3,957** |

| Requested rate | Byte-derived rate | Container bytes | Local frame bytes | Cold expert bytes | Cold amplification |
|---:|---:|---:|---:|---:|---:|
| 2.15 | 2.1500001130 | 7,608,730 | 1,267,439 | 1,271,535 | 1.0026916450x |
| 2.25 | 2.2500000000 | 7,962,624 | 1,326,422 | 1,330,514 | 1.0025695047x |
| 2.50 | 2.5000000000 | 8,847,360 | 1,473,878 | 1,477,970 | 1.0023125543x |

Cold reads include the complete 4,092–4,096 byte global manifest every time;
a cached reader is smaller still. These values are below `2x`, but they are a
layout ledger, not an emitted codec. In particular, the information oracle
grants lossless gauge coordinates, while the concrete ledger proposes FP32;
its finite-precision distortion has not been scored. This qualification cannot
rescue the branch because the strictly stronger free-side oracle already fails.

## Reproduction

The experiment is CPU-only. On the source host:

```bash
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 nice -n 15 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/invariant_manifold_oracle/gauge_coupled_polar_oracle.py \
  --root /workspace/INT2__compression \
  --output /workspace/INT2__compression/INT2_Q_C/research/invariant_manifold_oracle/result.json \
  --gaussian-replicates 1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 15 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/invariant_manifold_oracle/verify_result.py \
  --root /workspace/INT2__compression \
  --result INT2_Q_C/research/invariant_manifold_oracle/result.json \
  --output INT2_Q_C/research/invariant_manifold_oracle/verification.json
```

The independent verifier rehashed all eighteen BF16 sources, directly formed
the selected inverse-gauge residual matrices in original source space,
reconstructed every serialized manifold dimension, repeated the
reverse-waterfill arithmetic without importing the experiment, checked the
Gaussian `F/s` arithmetic, and rebuilt every byte/read ledger. It passes.

Artifacts:

- `gauge_coupled_polar_oracle.py` — source-locked CPU experiment
- `result.json` — complete 767-rank curves, selections, controls, and ledger
- `verify_result.py` — independent arithmetic/source-binding verifier
- `verification.json` — passing verification receipt with source rehashing
- `ARTIFACT_HASHES.json` — frozen artifact identities

## Claim boundary

This is an early-kill architecture result, not a compressed checkpoint. No
actual encoded MSE is claimed. It rules out this specific exact-gauge coupled
polar family under assumptions more favourable than a realizable PTQ codec;
it does not prove that every possible source model fails.
