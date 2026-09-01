# Rate-relative Qwen research checkpoint — 2026-09-01

## Status

This checkpoint preserves five completed architecture gates produced after the
STRATA expert-affine MoE locality milestone. It is a research checkpoint, not
a claim that the final 20%-below-Gaussian target has been reached.

The exact acceptance rule remains

```text
2.15 <= R_actual <= 2.5 bpw
F = MSE * 2^(2 R_actual) <= 0.8
s = -0.5 log2(F) >= 0.16096404744368115 bpw
maximum cold compressed read per routed expert < 2x
```

Every metadata bit, model table, frame, padding region, and source-derived
side stream belongs in `R_actual`. Read amplification is external compressed-
object traffic; it does not yet include a fused decoder/GEMM's scratch or HBM
traffic.

## Starting point and remaining gap

The already published expert-affine container remains the finite-code
baseline:

| Quantity | Audited result |
|---|---:|
| Physical rate | `2.5 bpw` exactly |
| Independently decoded relative MSE | `0.030902167403153148` |
| `F` | `0.9888693569009007` |
| `s` | `0.008074080480766676 bpw` |
| Worst cold 4-KiB expert read | `1.1694444444444445x` |
| Container SHA-256 | `4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b` |

The finite baseline still needs `0.15288996696291447 bpw` of source-specific
rate advantage. The strongest previously completed honest ideal-RD composite
has `F=0.9363976210`, `s=0.0474034129`, and `1.00278x` reads; it still needs
`0.1135606346 bpw` and has no corresponding finite serialized codec.

## Completed gates in this checkpoint

| Gate | Strongest valid result | Cold/page read | Decision |
|---|---:|---:|---|
| Shared input/output subspace | `F=0.9981959259638405`, `s=0.001302539625239886` | `1.1322916667x` | Hard kill |
| Tier-A initialization anchor | corrected validation capture `-0.0007686975`; optimistic `+2 SE=0.0013806933` | `1.1694907936x` | Hard kill for 56 sealed keys |
| Haar/ACG orientation entropy | charged optimistic `s=-0.0001247286`; nested need `0.1135606346` | `1.005556x` | Hard kill |
| Exact sparse-tail peeling | complete-grid `F>=0.979822670284832`; exhibited `F=0.9798226742103275` | `1.0392171224x` | Hard kill |
| Dual model-axis polar | raw `s=0.1087553446`; controlled excess upper `0.0192812045` | about `1.00034x` | Hard kill; generic polar leakage |

### Decoder-visible shared subspace

The gate learns joint-role and role-specific left, right, and two-sided bases
from twelve Layer-15 auxiliary experts, then evaluates four untouched whole
experts. The pinned 18-matrix panel is never opened. It evaluates 160 geometry
cells and 480 rate rows with a CuPy SVD implementation, exact basis arithmetic,
and an FP16-sized physical basis charge.

The best row is a role-specific right rank-16 basis at 2.5 bpw. It captures
`1.38593228%` of Up energy and `2.66697938%` of Down energy using `0.78125%`
of the coordinate dimensions, but its one-waterfill result is only
`F=0.9981959259638405`. The independent verifier rehashes 32 sources and
reselects all 480 rows.

Evidence: [`research/shared_subspace_gate/`](../research/shared_subspace_gate/)
and result SHA-256
`6a756c26e05926a2468bcc2a20086d7efbb2273e0f078a9294c5c002f2f7734f`.

### Tier-A initialization-anchor oracle

This auxiliary, pinned-panel-free gate tests whether final weights retain a
decoder-reproducible component of a plausible Hugging Face v4.51 initializer.
The candidate set was frozen before payload access:

- four stream scopes;
- seeds `0,1,42,1234,2023,2024,3407`;
- FP32-to-BF16 and direct-BF16 CUDA paths; and
- one globally selected key, with matched-Gaussian and permuted-anchor search
  controls.

The best source key is `hf451_layer_reset`, seed 0, FP32-to-BF16, but its
untouched validation capture is negative. Its `+2 SE` corrected upper bound
is `0.001380693345828353`, compared with `0.14580061597878702` required even
for a favourable ideal-composite nesting. The result does not identify or
make a claim about Qwen's undisclosed production framework or seed.

The independent verifier hashes all 31 eligible payloads, never opens the one
excluded payload, and confirms zero pinned-panel access. The complete result
and receipt are packaged in
[`research/initialization_anchor_oracle/`](../research/initialization_anchor_oracle/).
Result SHA-256:
`6ef38ff14f69ab02caf0e48ad37e5dbc3dfa9ebe7ba5e663f85080114e0f828d`.

### Haar/Stiefel orientation entropy

This gate canonicalizes raw QR/Householder Stiefel charts and fits a diagonal
angular-central-Gaussian model with leave-one-whole-expert-out folds over 16
auxiliary experts. A matched Haar/Gaussian control prevents generic chart
effects from being called Qwen structure. The model contains 256 FP16 values
(512 bytes).

The raw Qwen likelihood signal is only `7.874975147e-6 bpw`. Once its table is
charged, the optimistic upper advantage is `-0.0001247285971 bpw`, versus
`0.113560634568 bpw` needed beyond the prior composite. The pinned panel is not
opened. The independent verifier passes 20 binding/arithmetic/layout checks.

Evidence: [`research/haar_manifold_entropy/`](../research/haar_manifold_entropy/)
and result SHA-256
`ca1fd877e1ec206658a6aa68906d4fb217b9d85425d83e73775839f8d83c683c`.

### Exact charged sparse-tail peeling

For each matrix this family peels a stable top-absolute BF16 support, charges
`ceil(log2 binomial(N,k))` support bits, and losslessly codes the selected BF16
words with the best of four self-describing literal/Huffman modes. It then
rebuilds one residual decomposition using either raw roles or seven
support-pattern role KLTs and performs one ideal Gaussian waterfill.

A simple coordinate search is not used as the kill proof. The final Lagrange
dual enumerates all `20^3=8,000` three-role choices per expert for both
residual geometries and certifies all `20^18` panel configurations at every
tested rate. At 2.5 bpw:

```text
weakest certified lower F = 0.979822670284832
best exhibited F          = 0.9798226742103275
ideal relative MSE        = 0.030619458569072736
maximum cold read         = 1.0392171223958333x
```

The winning row peels only 93 of 28,311,552 weights. An illegal diagnostic
that reveals the selected values and bases for free reaches `F=0.6960`; the
gap shows that transmitting exact tail values, not support locality, kills
this architecture.

Evidence: [`research/tail_peeling_composite/`](../research/tail_peeling_composite/),
result SHA-256
`f2c3c1006d274d9025c1a61e3da9549bf22cd61bc333457c8198a34c538e2cf2`,
and internal lock
`f7d38fb4f67b77c5af577d8123286e4b26ded8b7f387f05a91f8b37faa85f547`.

### Dual model-axis polar red team

The tall stack

```text
X = [Gate; Up; Down.T] in R^(2304 x 2048)
```

initially appeared much stronger than prior gates. A source-adaptive polar
rank/window oracle reported `F=0.860048134470071` and
`s=0.10875534461889687`. It could not be promoted because an actual iid
Gaussian codec cannot beat the exact Gaussian rate-distortion function.

The frozen red team therefore repeats the identical rank/window search and
rate ledger on three moment-matched iid Gaussian panels plus an independent
Marchenko--Pastur `(8/9)` continuum:

| Statistic | `s` (bpw) |
|---|---:|
| Raw Qwen oracle | `0.10875534461889687` |
| Gaussian replica mean | `0.08955277315858347` |
| Marchenko--Pastur null | `0.08959477631590239` |
| Favourable source-specific excess upper | `0.019281204467028926` |

Thus 82.34% of the apparent gain is generic Wishart/polar-coordinate
metric/Jacobian leakage. Even granting the entire prior role-plus-horizontal
polar score and assuming zero overlap yields only `s=0.06668461734282799`,
short by `0.09427943010085316 bpw`. Role KLT also leaves the dual Gram exactly
invariant, so naively adding the raw scores is structural double counting.

The authorized CuPy control took 149.670186 seconds. An independent verifier
passes 15 spectrum/search/waterfill/control/nesting checks.

Evidence: [`research/dual_polar_oracle/`](../research/dual_polar_oracle/),
control result SHA-256
`4cc8c755e027475f3909d11069662ae2a5e40bbd665e65f6d7f32ea90494763d`,
and verification SHA-256
`9a8ed58c23c8434e346f11bc6ff4de10c0a2591cd34893cfe1ce558f001c611f`.

## Verification performed for this checkpoint

Local CPU unit suites pass:

```text
Tier-A initialization anchor: 13/13
Haar-manifold entropy:         8/8
Dual-polar red team:           5/5
Sparse-tail composite:        10/10
```

The source-holding RunPod independently verified the shared-subspace result,
the Tier-A result, the Haar result, the full sparse-tail source/certificate,
and the dual-polar control. Five nonrecursive `ARTIFACT_HASHES.json` manifests
bind 43 checkpoint files; rehashing all 43 files reproduces every recorded
length and SHA-256.

The production CuPy interpreter is

```text
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python
```

Exact reproduction commands and source-firewall rules are in each branch's
README. The initialization result was copied byte-for-byte from its sealed
RunPod output directory into the repository before this checkpoint.

## Active next experiment, deliberately excluded from the sealed result set

The highest-upside remaining hypothesis is a training-initializer residual:
if a decoder can regenerate the correct original Philox stream, the trained
weight may retain a large anchor component at essentially zero read cost.

Tier B expands the finite procedural family to 37,748,736 logical MCore keys.
The end-to-end seed rule is taken from the same immutable MCore commit:

```text
CLI base seed + 100*PP rank + 1024 + 100*EP rank + ETP rank.
```

For global Layer 15, PP sizes 1 and 2 are exactly equivalent under the locked
partition; PP4 and PP8 are distinct. The compute map therefore contains
28,311,552 distinct anchors while retaining and charging all logical ordinals.
The search uses a preregistered two-stage CuPy cascade, 32 matched null
searches, four untouched whole-expert folds, and no pinned-panel access. Its
total cap is 17,782,179,840 generated normal values. Eighty physical metadata
bytes would keep the projected cold read near `1.16946x`.

This active directory is not part of the present checkpoint because source-
free parity/calibration and the payload run are not yet complete. A positive
auxiliary result would still require a separately frozen Gate-role test and a
finite residual-codec replay; a negative result would reject only its finite
procedural family.

In parallel, a compact asymmetric nonlinear hyperdecoder is being audited as
the other logically open class. It must be demonstrably distinct from the
completed shallow BiSCo, affine-flow, additive-VQ, and linear-subspace gates,
fit below the `<2x` cold-read budget, and beat an identically trained Gaussian
control on untouched whole experts before it may access the pinned panel.
