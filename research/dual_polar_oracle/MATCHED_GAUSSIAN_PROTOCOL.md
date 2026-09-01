# Frozen matched-Gaussian red-team protocol

Frozen after `result.json` was received, and before generating or decomposing
any Gaussian control matrix.

## Question

The dual polar oracle reports `F=0.860048134470071` and
`s=0.10875534461889687 bpw` for

```text
X = [Gate; Up; Down.T] in R^(2304 x 2048).
```

An iid tall Gaussian matrix has a broad Marchenko--Pastur singular spectrum.
Treating its source-adaptive polar manifold and normal as independent Euclidean
Gaussian components can therefore appear to beat the scalar Gaussian RD limit
even though that limit is exact.  This diagnostic measures that null gain with
the identical rank/window search and rate ledger.

## Frozen controls

- Use three deterministic replicas, seeds derived from
  `SHA256("dual-polar-control-v1:26090131:replica:layer:expert:role")`.
- For each of the 18 authenticated source roles independently, draw iid
  Gaussian values, remove their finite-sample mean, rescale to the source's
  exact centered FP64 energy, add the source mean, and round to FP32.
- Canonicalize Down by transposition and stack roles exactly as the source
  oracle does.
- Apply the same full `2304 x 2048` singular-value decomposition, every
  `k=0,...,2046` rank, every contiguous unmodelled spectrum window, six-expert
  coordinate descent, side ledger, and panel reverse waterfill.
- Score `R in {2.15, 2.30, 2.50}`.  No control-specific rank restriction,
  spectrum smoothing, or target selection is allowed.
- Also compute a source-independent continuum Marchenko--Pastur control at
  aspect `2048/2304=8/9`, discretized at 2,048 midpoint quantiles and rescaled
  to each expert's energy.  This is an analytic cross-check, not a replacement
  for the finite Gaussian replicas.

The parent source result, source lock, 18 source hashes, this protocol, and the
executing script are bound by SHA-256.

## Frozen verdict arithmetic

Let `s_Q` be the source oracle score and let `s_C` be the mean matched-control
score at 2.15 bpw.  Report:

```text
generic_fraction = s_C / s_Q
excess_s          = s_Q - s_C
control_floor     = min(s_MP, s_C - 3*SE_replica)
excess_upper      = max(0, s_Q - control_floor)
```

The control-subtracted quantity is a diagnostic upper opportunity, not an
achievable-code proof.  The nonlinear polar coordinate metric/Jacobian error
need not cancel perfectly.

For the most favourable nesting audit, grant the entire existing charged
role-plus-horizontal-polar score `s_H=0.047403412875799064 bpw`, assume zero
overlap with `excess_upper`, and form

```text
s_union_upper = s_H + excess_upper.
```

This deliberately generous union survives only if
`s_union_upper >= 0.16096404744368115`.  Raw `s_Q+s_H` is also shown, but is
explicitly invalid because both polar oracles may credit the same generic
Wishart geometry.  A survivor would require a direct intrinsic joint
decomposition and one waterfill; scalar addition is never accepted as a
result.

## Runtime rule

The diagnostic may use CuPy only after the parent confirms no concurrent GPU
job.  Until then, only source-independent Marchenko--Pastur arithmetic and
read-only inspection are permitted.
