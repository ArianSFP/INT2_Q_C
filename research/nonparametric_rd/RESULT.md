# Nonparametric BA oracle: Qwen scalar and short-vector verdict

## Decision

Kill this branch. On the sealed 18-matrix Qwen panel, neither scalar
non-Gaussianity nor adjacent 2-D/4-D dependence provides a positive held-out
rate advantage after an intentionally favorable matched-Gaussian calibration.

The target requires

```text
s_required = -0.5 log2(0.8) = 0.16096404744368115 bpw
F_required = 2^(-2 s_required) = 0.8.
```

The best panel-aggregate free-side result is scalar raw coefficients at 2.15
bpw:

```text
s = -0.008716492144347285 bpw
F = 1.0121569258452994
```

After charging even the small scalar model, `F=1.0123742243765088`. Thus the
source was slightly *harder*, not easier, than its moment-matched Gaussian
control under the same finite test-channel apparatus.

For a deliberately extreme uncertainty allowance, take the largest gain of
any one held-out expert/rate/branch (`0.002543053253034128` bpw) and add another
`0.005` bpw for numerical and support error. This gives only

```text
s_optimistic = 0.007543053253034128 bpw
F_optimistic = 0.9895975910330618
shortfall     = 0.15342099419064703 bpw.
```

This upper diagnostic is still more than twenty times too small in rate-gain
terms. A conventional denser BA sweep is therefore a certain dead end.

## What was tested

- Literal 28,311,552-weight, 18-source plan sealed by lock
  `99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`.
- Raw coordinates `{Gate, Up, Down.T}` and the current header's literal FP32
  `{Gate, XKLT0, XKLT1}` coordinates.
- Whole-matrix mean/RMS normalization, with its transmission explicitly
  charged in the realizable ledger.
- Scalar, adjacent 2-D, and adjacent 4-D vectors.
- Six leave-one-expert-triplet-out folds. The reconstruction support,
  coordinate rotation, and BA output prior are learned using only the other
  five experts for the corresponding component.
- Held-out rate
  `E KL(P_beta(reconstruction|x) || q_train(reconstruction)) / d`; the output
  prior is not refit on the test matrix.
- A deterministic moment-matched Gaussian control with the same sample count,
  support geometry, beta grid, and stopping rule. Dividing by this control
  credits away *all* finite-support and solver loss, making the promotion score
  optimistic.

This is not another GGD, kurtosis, histogram-KL, variance-stratum, or
shape-class likelihood screen. It directly solves the finite empirical
stochastic test channel.

## Results at 2.15 physical bpw

| Coordinates | Dimension | Free-side `s` (bpw) | Free-side `F` | Charged `F` | Model bpw | Conservative expert read amp at 2.15 bpw |
|---|---:|---:|---:|---:|---:|---:|
| raw | 1 | -0.00871649 | 1.01215693 | 1.01237422 | 0.00015485 | 1.11154325x |
| XKLT | 1 | -0.00895555 | 1.01249241 | 1.01270978 | 0.00015485 | 1.11154325x |
| raw | 2 | -0.03743492 | 1.05326600 | 1.05441859 | 0.00078894 | 1.11331279x |
| XKLT | 2 | -0.03468575 | 1.04925949 | 1.05040769 | 0.00078894 | 1.11331279x |
| raw | 4 | -0.02902192 | 1.04105323 | 1.04886909 | 0.00539539 | 1.12616803x |
| XKLT | 4 | -0.04513764 | 1.06457329 | 1.07256573 | 0.00539539 | 1.12616803x |

The 4-D physical ledger is the largest: three full FP16 reconstruction tables,
three uint16 BA-frequency tables, three FP32 rotations, all 18 FP32 mean/RMS
pairs, and a global choice header total 152,752 bits. Reading that common table
cold for every expert increases the expert-affine `10/9` coefficient-volume
ratio to `1.1240600585937501x` at 2.5 bpw and `1.1261680267885983x` at the
worst permitted rate of 2.15 bpw, safely below 2x throughout. No per-vector or
per-group label stream was granted for free.

## Reproduction and verification

The executed script is frozen at SHA-256
`b62f2dfc466c2be60f8c3de7e368645233fd573323404e53b76a63b716b2127f`.
The two early-kill runs were:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python rd_nonparametric_oracle/nonparametric_ba_oracle.py \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --output rd_nonparametric_oracle/qwen_scalar_quick.json \
  --dimensions 1 --samples 1:4096 --levels 1:32 \
  --betas 5,8,12,18,28,45 --max-iterations 20 --tolerance 1e-5 --chunk 1024

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python rd_nonparametric_oracle/nonparametric_ba_oracle.py \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --output rd_nonparametric_oracle/qwen_joint_quick.json \
  --dimensions 2,4 --samples 2:1024,4:512 --levels 2:12,4:5 \
  --betas 5,10,18,32,64 --max-iterations 10 --tolerance 1e-4 --chunk 512
```

Verify both internal result seals, source bindings, script hash, branch
coverage, gain/F identities, side ledgers, and read gate:

```bash
python rd_nonparametric_oracle/verify_and_summarize.py \
  --script rd_nonparametric_oracle/nonparametric_ba_oracle.py \
  --scalar rd_nonparametric_oracle/qwen_scalar_quick.json \
  --joint rd_nonparametric_oracle/qwen_joint_quick.json \
  --output rd_nonparametric_oracle/qwen_nonparametric_ba_summary.json
```

Evidence hashes:

```text
39aed315c92a4e1a9fab4e9ef9a508cc6d581e013936f9c17de3745eebd4a159  qwen_scalar_quick.json
869ad529bb79e8f846411ba8e98ca6d2a7a33eb052405ea616c63cc4c65c652c  qwen_joint_quick.json
```

The summary JSON contains all 18 literal source hashes, the plan/header hashes,
exact `F` calculations, and the corrected per-rate six-fold uncertainty. The
large result JSONs retain every beta point, convergence record, covariance,
sample hash, fold model hash, and source-energy weight.

## Claim boundary

The hard kill applies to stationary scalar and adjacent short-vector
nonparametric channels under the tested raw/XKLT normalizations. It is not a
universal converse for long-range deterministic structure, semantic
equivalences, or a functional rather than source-domain objective. Those are
the only classes broad enough to remain plausible after this result.
