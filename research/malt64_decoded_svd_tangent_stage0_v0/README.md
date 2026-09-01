# MALT64 stage-0 decoded-SVD tangent screen

MALT means **M**anifold-**A**daptive **L**ocal **T**angent.  This auxiliary
screen tests a concrete decoder-conditioned refinement that is not covered by
the fixed-basis residual probes.

For every exact 4,096-value block from the independently decoded STRATA
expert-affine checkpoint, the decoder reshapes its already available coarse
reconstruction as `64 x 64`, computes its leading rank-three left/right
singular subspaces, and defines the rank-three matrix-manifold tangent

```text
T_Y = { U A^T + B V^T }.
```

Its dimension is `3*(64+64-3)=375`.  A future finite cell would use 384
target bits per block, so stage 0 grants every block arbitrary real tangent
coefficients.  This projection strictly dominates every finite correction
whose reconstruction stays in `T_Y`.  The remaining nine refinement bits are
ignored; that 2.4% rank shortfall cannot explain a large miss, but the claim is
scoped to the frozen rank-three tangent cell.

The physical planning cell is the exact TACTIC-style ledger:

- coarse payload: `2.3984375 bpw`;
- 384 target bits per 4,096 values: `0.09375 bpw`;
- all model/header bytes: `0.0078125 bpw`;
- total: exactly `2.5 bpw`;
- conservative cold page read: `73/72 = 1.0138888888888888x`.

Under the deliberately favorable assumption that the measured finite base
factor transfers unchanged to the lower coarse rate, the tangent must capture
at least `0.2972443434920543` of the coarse error.  The null rank share is only
`375/4096 = 0.091552734375`.

The program authenticates and reconstructs all six pinned expert triplets
from the exact independent-decode scratch, uses every one of the 6,912
contiguous blocks, performs FP64 batched SVD/projection in CuPy, and reports a
delete-one-expert jackknife upper-three-SE bound.  It checks that its total
coarse SSE reproduces `500.39553685426534` before making a decision.  If the
upper bound is below the planning threshold it stops before controls, finite
coefficient search, validation, or any new encode.

RunPod command:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B stage0_screen.py \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --output /var/tmp/malt64_stage0_result.json
```

This is an architecture-scoped feasibility screen, not a compressed artifact
or a universal converse for nonlinear residual coding.
