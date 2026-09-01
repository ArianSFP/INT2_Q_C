# Dual model-axis polar oracle: matched-Gaussian red-team

## Verdict

**Hard kill.**  The source oracle's favourable `F=0.8600481345` is real
arithmetic for its declared component waterfill, but most of that apparent
advantage is generic rectangular Gaussian polar geometry rather than Qwen
structure.  It is not evidence for a codec that beats the Gaussian RD limit.

At `R=2.15`:

| Quantity | `F` | `s=-0.5 log2(F)` | Relative MSE |
|---|---:|---:|---:|
| Qwen source oracle | `0.8600481345` | `0.1087553446` | `0.0436610099` |
| Matched Gaussian, replica 0 | `0.8832995817` | `0.0895126333` | `0.0448413876` |
| Matched Gaussian, replica 1 | `0.8831901049` | `0.0896020432` | `0.0448358299` |
| Matched Gaussian, replica 2 | `0.8832616108` | `0.0895436430` | `0.0448394600` |
| Continuum Marchenko--Pastur null | `0.8831990022` | `0.0895947763` | `0.0448362816` |
| Required | `<=0.8` | `>=0.1609640474` | `<=0.0406126198` |

The matched-control mean is `s=0.0895527732`; it explains **82.3433%** of
the source score.  The Qwen-minus-control mean is only
`0.0192025715 bpw`.  Giving Qwen the most favourable of the analytic null and
the matched-control three-standard-error lower bound raises that residual only
to `0.0192812045 bpw`.

## Frozen diagnostic

`MATCHED_GAUSSIAN_PROTOCOL.md` was frozen before any control matrix was
generated.  For every authenticated Gate, Up, and Down source independently,
the diagnostic generated three deterministic Gaussian replicas matching its
mean and centered energy.  Each control then used exactly the source oracle's:

- `2304 x 2048` canonical stack;
- full singular-value decomposition;
- all 2,047 ranks and all contiguous unmodelled spectral windows;
- adaptive six-expert coordinate descent;
- side-bit ledger; and
- panel-wide reverse waterfill at 2.15, 2.30, and 2.50 bpw.

The independent Marchenko--Pastur calculation used aspect
`2048/2304 = 8/9`, 2,048 midpoint quantiles, and the source expert energies.
Its normalized singular support is

```text
1 - sqrt(8/9) = 0.0571909584
1 + sqrt(8/9) = 1.9428090416.
```

That broad iid-Gaussian bulk is the same phenomenon selected by the source
oracle: every winning Qwen window begins at the bottom of the spectrum and
collapses approximately 1,500 low modes to one scale.

## Why `F<1` on iid Gaussian is a red flag

For an iid Gaussian source under squared error, the Gaussian RD function is
exact.  An invertible change of coordinates cannot genuinely improve it.
Nevertheless the ideal polar component calculation reports
`F=0.8831990022` on the iid Marchenko--Pastur null.

The issue is not a source-file leak.  It is a nonlinear coordinate-metric and
measure omission.  For `X=QH`, tangent distortion is induced by

```text
dX = dQ H + Q dH,
```

so Stiefel directions are weighted by the source-specific `H`.  The polar
volume element also contains singular-value terms equivalent to

```text
J(H) proportional to
  product_i s_i^(2304-2048) * product_(i<j) (s_i + s_j),
```

when `dH` is ordinary symmetric-matrix measure.  Counting manifold and normal
dimensions, assigning each its ambient projection energy, and applying an
isotropic Euclidean Gaussian waterfill omits this metric/Jacobian.  The
Marchenko--Pastur spread is consequently credited as free variance structure.

The source-selected window index is also not charged in the stage-one side
ledger, although this does not drive the winning result: every selected Qwen
and control window is the canonical bottom window (`window_start=0`).  Rank
labels are charged, and the compressed-object read ledger remains below 2x.

## Non-double-counting nesting audit

The raw dual score is short of the required `s` by
`0.0522087028 bpw`.  The existing charged role-plus-horizontal-polar oracle has
`s=0.0474034129`.  Even the invalid scalar sum is only

```text
0.1087553446 + 0.0474034129 = 0.1561587575 bpw,
```

still `0.0048052899 bpw` short.

More importantly, adding these scores double counts structure.  A 3x3 role
KLT left-multiplies the stack by `U role-kron I_768`, so

```text
X' = (U kron I_768) X
X'.T X' = X.T X.
```

The dual symmetric factor is therefore exactly invariant to role KLT.  The
horizontal and dual polar models also both credit dispersion derived from the
same source singular geometry, including the same Gaussian/Wishart effect.

For an intentionally favourable upper screen, retain the entire uncorrected
role-plus-horizontal score, grant zero overlap, and add only the dual
Qwen-specific three-SE residual:

```text
s_union_upper = 0.0474034129 + 0.0192812045
              = 0.0666846173 bpw.
```

This remains `0.0942794301 bpw` below the requirement.  Thus neither the raw
sum nor a control-adjusted, zero-overlap union plausibly closes the target.

## Only defensible joint architecture

The rigorous version would be an **intrinsic two-sided polar quotient codec**,
not two added oracles:

1. Form the role stack once and factor `X=Q_c H_c`.
2. Entropy-code `H_c` with its polar volume term and code `Q_c` under the
   `H_c`-weighted tangent distortion metric.
3. Model horizontal/role structure only conditionally inside `Q_c` or its
   exact normal bundle.  Do not encode or score singular-value dispersion a
   second time.
4. Remove the shared scalar/orthogonal gauge once, charge all chart and model
   indices once, and perform one joint waterfill.
5. Require the identical implementation to return `F>=1` on iid Gaussian
   controls, within finite-sample error, before inspecting Qwen benefit.

That architecture is well-defined, expert-local, and compatible with a
near-1x contiguous frame.  It is not promoted: the generous residual bound is
far too small to justify its finite codec or another GPU search.

## Reproduction

The single authorized CuPy control job ran for `149.670186` seconds on the
provided RTX 5090:

```bash
cd /workspace/INT2__compression
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/dual_polar_oracle/dual_polar_matched_gaussian.py \
  --source-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --source-result INT2_Q_C/research/dual_polar_oracle/result.json \
  --source-script INT2_Q_C/research/dual_polar_oracle/dual_polar_oracle.py \
  --protocol INT2_Q_C/research/dual_polar_oracle/MATCHED_GAUSSIAN_PROTOCOL.md \
  --composite-result INT2_Q_C/research/composite_superoracle/result.json \
  --output INT2_Q_C/research/dual_polar_oracle/matched_gaussian_result.json \
  --backend cupy --replicas 3 --mp-grid-points 1000001
```

The CPU-only independent verifier rebuilds all serialized spectra, full rank
curves, common-rank global optima, adaptive coordinate-wise optima,
waterfills, Marchenko--Pastur quantiles, control statistics, and nesting
arithmetic:

```bash
cd /workspace/INT2__compression
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/dual_polar_oracle/verify_matched_gaussian.py \
  --result INT2_Q_C/research/dual_polar_oracle/matched_gaussian_result.json \
  --source-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --source-result INT2_Q_C/research/dual_polar_oracle/result.json \
  --source-script INT2_Q_C/research/dual_polar_oracle/dual_polar_oracle.py \
  --protocol INT2_Q_C/research/dual_polar_oracle/MATCHED_GAUSSIAN_PROTOCOL.md \
  --control-script INT2_Q_C/research/dual_polar_oracle/dual_polar_matched_gaussian.py \
  --composite-result INT2_Q_C/research/composite_superoracle/result.json \
  --output INT2_Q_C/research/dual_polar_oracle/matched_gaussian_verification.json
```

It passed all 15 checks.  No second GPU diagnostic was launched.

## Claim boundary

This result kills the current dual-polar ideal-RD score and naive nesting as
evidence for the 20%-below-Gaussian target.  It does not prove that every
intrinsic two-sided manifold codec is impossible.  Any future claim must
include the correct induced metric/measure, one non-overlapping rate ledger,
an emitted finite stream, source-domain MSE, and a matched iid-Gaussian null.
