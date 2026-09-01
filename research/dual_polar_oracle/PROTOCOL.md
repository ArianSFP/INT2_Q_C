# Dual (model-axis) polar oracle protocol

Frozen before executing this geometry on the pinned panel.

For each expert, canonicalize all roles to `768 x 2048` and form the tall
matrix

```text
X = [Gate; Up; Down.T] in R^(2304 x 2048).
```

This is the untested dual of the existing coupled polar oracle, which forms
`[Gate, Up, Down.T] in R^(768 x 6144)` and therefore models the hidden-neuron
axis.  The dual construction models the common 2,048-dimensional model axis.

Compute `X = Q H`, `Q^T Q = I`, and approximate the symmetric factor as
`H_hat = c I + A_k`. For every feasible rank `k`, exhaust all contiguous
unmodelled singular-value windows; this is the exact least-squares optimum for
that rank. The model and residual are Frobenius-orthogonal. Charge their exact
manifold/normal dimensions and perform one panel-wide ideal Gaussian reverse
waterfill after deducting framing/rank bits.

Stage 1 is deliberately favorable: continuous coordinates, exact charts,
exact source-selected ranks, and asymptotic component codes. If its own
absolute score is above `F=0.8`, the family is a certain dead end and no
Gaussian control or finite encode is run. If it crosses `0.8`, run identical
moment-matched Gaussian controls and require a control-adjusted advantage
before any finite implementation.

Requirements remain `2.15 <= R <= 2.5`, `F <= 0.8`, and cold compressed
expert reads `<2x`. The pinned source lock and all 18 source hashes must pass.

