# Rate-relative breakthrough search

## Exact objective

The research target is not “best MSE at any rate.”  For the artifact's own
physical rate `R`, with `2.15 <= R <= 2.5`, it must satisfy

```text
MSE <= 0.8 * 2^(-2R).
```

The compressed bytes needed for one routed expert must be less than twice one
expert's equal physical share.  Results close to `1x` are preferred, and a
higher read cost below `2x` is acceptable only for a material MSE benefit.

It is useful to write

```text
F = MSE * 2^(2R)
s = -0.5 * log2(F).
```

Success is `F <= 0.8`, equivalently
`s >= 0.16096404744368115 bpw`.  This identity lets an optimistic structural
oracle be killed before a multi-hour GPU encode when it cannot possibly
supply the required advantage.

At the endpoints:

| Physical rate | Gaussian reference | Required MSE |
|---:|---:|---:|
| 2.15 bpw | `0.050765774772264724` | `0.04061261981781178` |
| 2.50 bpw | `0.03125` | `0.025` |

## Locality checkpoint and remaining gap

The expert-affine checkpoint is a separate first milestone.  It replaces the
globally interleaved fourteen-block layout with twelve private N21 streams and
three paired N20 tails.  One expert reads two private streams and one shared
tail, giving a structural coefficient-volume ratio of `10/9 = 1.111111...x`.
Its physical container is exactly 2.5 bpw.

The sealed allocation projected MSE `0.03090139432980219`; projection was not
treated as an achieved result. Independent causal decode/re-encode and scoring
against all eighteen original BF16 files subsequently measured MSE
`0.030902167403153148`, `F=0.9888693569009007`, and only
`s=0.008074080480766676 bpw`. The remaining rate-equivalent gap is
`0.15288996696291447 bpw`. The audited worst cold 4-KiB expert read is
`1.1694444444444445x`.

Even an oracle continuous waterfill over all current groups reaches only about
`0.03013248` at 2.5 bpw (`F=0.96423936`, `s=0.02626839 bpw`).  Therefore profile
retuning or a larger reserve cannot deliver the target; a new source model is
required.

## Search discipline

Every branch follows the same order:

1. authenticate the pinned six-expert/eighteen-matrix panel;
2. test an intentionally favorable, often free-side-information oracle;
3. convert its best possible benefit to `s` or `F`;
4. charge tables, transforms, labels, framing, and cold expert reads;
5. run a full CuPy encoder only if the favorable oracle is near the gate; and
6. require a byte-derived rate plus independent original-source scoring before
   calling a result achieved.

The oracle rows below are not additive.  Several probe overlapping forms of
the same weak dependency, and some are information bounds rather than
operational reconstructions.

## Completed early-kill results

| Family | Most favorable observed opportunity | Read implication | Decision |
|---|---:|---:|---|
| Nonparametric scalar/2-D/4-D empirical RD | best aggregate `s=-0.00871649`; extreme single-fold-plus-allowance `s=0.00754305` | worst charged `1.126169x` at 2.15 bpw | Kill |
| Additive residual VQ, d=8/16/32 | every matched gain negative; two-SE-plus-allowance `s=0.00553294` | `1.001285x`--`1.020885x` | Kill |
| ICA / projection pursuit, d=8/16/32/64 | free-side `s=0.00145369`; LOO-plus-allowance `s=0.00699315` | charged d64 `1.143815x` at 2.15 | Kill |
| Polar list / nearest-codeword search | complete large-N gap `s=0.00733`--`0.00782`; exact N16 rate-ignored gain `7.21%`--`12.66%` | unchanged expert-local layout | Kill |
| Pre-RHT companding / marginal negentropy | best cross-fitted net `s=-0.00162370` | small shared tables | Kill |
| Conditional scale hyperprior | net `s=0.00493470` | approximately `1x` warm | Kill |
| Nonlinear aligned semantic-role predictor | impossible free-side gain `3.05499%`; charged `s=0.01940678` | a realizable reference stream is unfavorable | Kill |
| SwiGLU neuron permutation/prediction | illegal many-to-one `s=0.00705925`; best legal `s=0.00536452` before side charge | best cold legal pair `2.00151x` | Kill |
| Local categorical/bitplane contexts | best legal `s=0.00008960`; leaky oracle `s=0.00370826` | at most about `1.01243x` | Kill |
| Shape-conditioned partial Hadamards | best charged `s=-0.00326034` | expert-local | Kill |
| PRISM covariance components | best optimistic cross-fit `F=0.97157253` | dense learned bases are too costly | Kill |
| Adaptive partial polar–Gram manifold | continuous/free-table `F=0.95049088`, `s=0.03662765` | charts and rank labels granted free | Kill |
| Tensor/Kronecker/TT/MPO structure | best matrix Kronecker proxy about `s=0.01848`; pooled separability about `s=0.00090` | second stream/table costs not justified | Kill |
| Post-decoder gain/polynomial correction | at most about `s=0.00005` | negligible metadata | Kill |
| Same-layer router side information | free adaptive rank-128 basis `F=0.99849179`, `s=0.00108877` | router granted resident/free | Kill |
| Exact SwiGLU gauge + joint role manifold | free adaptive `F=0.98740818`, `s=0.00914075`; matched Gaussian is better | charged frame `1.00231x`--`1.00269x` | Kill |
| PRG union of subspaces | leakage-safe `s=0.0014553`; optimistic physical `F=1.05360` | `1x` | Kill |
| NanoQuant binary factor | matched `s=0.0136368`, but actual codec `F=3.19136` at 2.5 | `1x` | Kill |
| External semantic dictionaries | at most `1.0463%` energy explained vs about `20.9%` required | one atom can approach `1.97x`; two exceed the gate | Kill |
| 2-D spectral scale fields | impossible free IPF `F=0.94972524`; best charged `F=0.97829416` | all candidates below `1.04528x` | Kill |
| Nested role + polar composite | best charged `F=0.93639762`, `s=0.04740341` | `1.00278x` | Kill; normal-field predictor isolated |
| Polar normal-field predictors | even exact continuous coefficients at an impossible one bit each give `F=0.86210289` | budget-fitting fields remain near `1.003x` | Kill |
| BiSCo shallow nonlinear binary decoder | independent FP64 `s_match=-0.00390485`; `D_Qwen=0.11020814` at the 512-update gate | analytic production ledger `1.020428x` at `2.250382 bpw` | Hard kill before pinned panel |

### Direct empirical rate-distortion oracle

The [nonparametric RD audit](nonparametric_rd/RESULT.md) solves cross-fitted
finite empirical test channels for raw and XKLT coordinates.  It tests scalar,
adjacent 2-D, and adjacent 4-D vectors and divides each result by an identically
configured moment-matched Gaussian control.  The best panel aggregate is
slightly worse than Gaussian.  Even the largest individual held-out fold plus
an extra `0.005 bpw` allowance leaves `0.1534209942 bpw` of the requirement.
The complete beta curves and verifier are retained in
[`nonparametric_rd/`](nonparametric_rd/).

The [additive-VQ audit](additive_vq/final_xklt/ADDITIVE_VQ_EARLY_KILL.md)
extends the empirical screen to 8-D, 16-D, and 32-D XKLT vectors with
role-conditioned binary, ternary, and quaternary additive residual
codebooks.  It uses six held-out experts, charges FP16 codebooks, indices,
scalars, tables, and framing, and calibrates every cell against an
independently fitted Gaussian control.  All raw matched advantages are
negative.  Its most generous uncertainty allowance reaches only 3.437% of
the required `s`, so a production CuPy fit was not justified.

The [ICA/projection-pursuit audit](ica_projection/RESULT.md) then asks whether
near-Gaussian scalar marginals conceal an independent non-Gaussian basis.
It tests KLT and symmetric FastICA contrasts through d=64.  Identity wins the
most favorable free-side screen.  In the stricter leave-one-expert-out,
rate-matched confirmation, even adding `0.005 bpw` to the best result leaves
`F=0.99035227`; serialized d64 side information makes the net gain negative.

### Encoder search gap

The [polar search audit](polar_search_gap/README.md) grants a hypothetical
encoder the entire measured finite-code loss of every frozen Qwen block.  That
reduces the projected 2.5-bpw MSE only to `0.0305683421`, not `0.025`.
An independent exact N16 test exhausts every constrained low-plane assignment
and solves the unrestricted nearest upper planes.  Even its rate-ignored
oracle stays below a 20% aggregate reduction, while production N20/N21 blocks
show only about a 1% operational gap.  Full-block list decoding was therefore
stopped before consuming GPU time.

### Marginal companding hypothesis

A signed Hadamard transform can make scalar marginals Gaussian while retaining
joint negentropy.  The [source-negentropy probe](source_negentropy_oracle.py)
therefore examines authenticated pre-RHT BF16 staging values.  It normalizes
by transmitted block RMS, then fits global, role, label, and role-by-label
tables with six leave-one-expert-out folds.  Each comparison Gaussian has the
same class mean and variance, preventing variance allocation from being
credited twice.

| Conditioning | Gross gain | Net after uint16 table |
|---|---:|---:|
| Global | `-0.0010449984` | **`-0.0016237021`** |
| Role | `-0.0015596650` | `-0.0032957761` |
| STRATA label | `-0.0051619262` | `-0.0097915558` |
| Role + label | `-0.0073200958` | `-0.0212089847` |

This rules out a componentwise compander.  It does not rule out a genuinely
high-dimensional learned code, which is why additive VQ and hidden-ICA probes
are evaluated separately.

### Conditional and semantic models

The [conditional hyperprior audit](conditional_hyperprior/README.md) tests 105
tile/width/step combinations with leave-layer-and-expert-out training.  Its
best physically charged gain is only `0.0049346951 bpw`, and 40 of 57 local
folds are negative.

The [nonlinear role oracle](nonlinear_semantic/) gives each predicted role the
other two roles losslessly—an impossible advantage—and counts all directions
at once.  It removes only `3.0549904%` of energy; after its small private table,
the optimistic equivalent advantage is `0.01940678 bpw`.

The [neuron-permutation oracle](NEURON_PERMUTATION_ORACLE.md) tests all 30
directed expert pairs and exact Hungarian assignments shared across Gate, Up,
and Down.  Even illegal many-to-one role-wise reuse supplies just 4.39% of the
required `s`.  A legal cold predictor also crosses the strict read ceiling
once its reference and mapping are charged.

### Local categorical and covariance models

The [bitplane/context audit](bitplane_context/agent_bitplane_context_report.md)
tests causal 2-D neighborhoods, semantic position, split sign/magnitude planes,
and cross-role categorical context.  Its strongest legal result reaches only
0.056% of the required rate advantage.

The [shape/PRISM probes](shape_strata/) test partial Hadamards, scale-invariant
shape classes, learned covariance spectra, KLT32/KLT64, and reverse waterfill.
The most optimistic PRISM result still has `F=0.97157`, far from `0.8`; a
single dense 2048-D FP16 basis would itself cost `2.37037 bpw` when amortized
over only these six experts.

The [Stiefel/Gram audit](stiefel_gram_oracle/README.md) explores a more radical
whole-matrix manifold.  It writes `W=HQ`, keeps the full row-Stiefel factor,
and models the symmetric polar factor as `H=cI+A_k`.  Every rank
`k=0,...,766` is solved exactly, followed by ideal 36-component
reverse-waterfilling.  The adaptive optimum uses ranks 179--241 and is granted
continuous coordinates plus free charts, tables, and labels.  Its ideal MSE
is still `0.02970284` at 2.5 bpw.  This is the strongest structural oracle in
the present search, but it supplies only 22.76% of the required equivalent
rate advantage.

### Decoder-visible and exact-symmetry side information

The [router-side-information audit](router_sideinfo_oracle/README.md) derives
three bases solely from the same-layer `128 x 2048` router, which is available
before an expert fetch. It grants exact basis arithmetic, the router bytes,
and source-leaky per-matrix mode selection for free. Its best rank-128
subspace captures `7.6683%` of energy with `6.25%` of dimensions; ideal
two-component waterfilling still gives only `F=0.99849179`. The independent
verifier rehashed all eighteen sources and six router byte ranges and checked
all 96 rate cells.

The [exact-gauge joint-manifold audit](invariant_manifold_oracle/README.md)
then quotients the legal SwiGLU action `U -> A U`, `D -> A^-1 D`, restores the
gauge under the exact source metric, and fits one coupled polar manifold to
Gate, Up, and Down. Even free continuous charts and source-adaptive ranks
reach only `F=0.98740818`; identically processed matched Gaussian triplets do
better, so the control-adjusted advantage is negative. Its explicit
expert-local side frame remains near `1.0025x` reads, but locality cannot
rescue the missing information gain.

### Procedural, binary-factor, and existing-weight codebooks

The [procedural-subspace package](procedural_subspace_oracle/README.md) closes
seed-selected PRG subspaces at `s=0.0014553`. Its initial SVD-tail ratio of
`0.75797` was rejected as a degrees-of-freedom leak: it granted a
1,567,728-DOF source-specific rank manifold and scored only the 5,136-DOF
discarded tail. A discrete NanoQuant-style binary factor achieves a small
matched Qwen/Gaussian advantage (`s=0.0136368`) but terrible absolute shaping
(`F=3.19136` at 2.5 bpw). LiftQuant's `Mq` decoder is algebraically contained
in the already-negative binary additive-VQ screen.

The [breakthrough red-team](breakthrough_redteam/README.md) also grants each
weight row its best atom from the complete embedding or attention dictionary.
The resulting `0.841%`--`1.046%` explained energy is essentially the random
extreme-value baseline and far below the roughly `20.6%`--`20.9%` needed.
NanoQuant's more favorable continuous-coefficient cut-factor screen peaks at
only about `s=0.0496` and falls to `s=0.04445` at its 2.5-bpw ledger; even
perfect Gaussian shaping would then be only 5.98% below the reference.

BiSCo's shared nonlinear binary-spherical decoder is not strictly contained
by these additive/linear screens. The preregistered shallow `d=16, h=64,
18+18` CuPy gate nevertheless stopped at update 512: independent state-backed
FP64 replay measured `D_Qwen=0.1102081376`, `D_Gaussian=0.1096131633`, and
`s_match=-0.0039048473`. All four untouched whole-expert folds were negative.
The physical deployment ledger was favorable (`2.250382 bpw`, `1.020428x`
cold reads), but locality cannot rescue a codec whose matched source advantage
is negative. The sealed run, serialized states, independent evaluator, and
per-matrix code/reconstruction hashes are retained in
[`bisco_raw_mse_oracle/`](bisco_raw_mse_oracle/). This kills only the frozen
shallow cell; the pinned panel was never opened by that branch.

### Long-range scale fields and a genuinely nested composite

The [spectral scale-field oracle](spectral_scale_field_oracle/README.md) gives
the decoder exact source-fitted row, column, and `16 x 32` tile energy fields
for free before ideal waterfilling. Even the impossible IPF union reaches only
`F=0.94972524`; charging its factors erases the gain. Four matched Gaussian
replicates reduce the Qwen-specific advantage to `s=0.01854135`. Read
amplification remains below `1.04528x` throughout, confirming that source
structure—not locality—is the bottleneck.

The [composite super-oracle](composite_superoracle/README.md) does not add
separately measured gains. It rebuilds authenticated residuals after a joint
role KLT and polar split, charges component dimensions and side bytes, and
performs one waterfill. Its best legal nesting gives `F=0.93639762`,
`s=0.04740341`, and MSE `0.02926243` at 2.5 bpw. Applying STRATA before local
polar decomposition is worse, demonstrating that the earlier gains overlap.

A deliberately illegal but diagnostic envelope reveals the exact polar
normal correction for free and reaches `F=0.66396845`. That side occupies
`1.54999 bpw` in FP16 but may consume only about `0.12025 bpw` after the
explicit header/rank charge while preserving `F<=0.8`. This is not a result;
it isolates a sharply budgeted next hypothesis: a held-out procedural
predictor for the polar normal field.

The [polar normal-field follow-up](polar_normal_predictor/README.md) closes
that clue for shared, analytic, router-derived, banded, DCT, and tiny implicit
families. A free identity band of width 128 appears to pass at `F=0.79189`,
but its FP16 field costs `0.92383 bpw`. Even granting every continuous
source-specific coefficient exact reconstruction for an impossible one-bit
ledger produces only `F=0.86210289`, above the target. The best genuinely
budget-fitting FP16 candidate is indistinguishable from the original charged
polar oracle. No GPU integration is justified.

## Relationship to recent PTQ architectures

Recent methods identify the same broad mechanisms being tested here:

- [AQLM](https://arxiv.org/abs/2401.06118) uses sums of learned codewords and
  reports its strongest practical regime near 2.5 bits per parameter.
- [Grouped Lattice Vector Quantization](https://arxiv.org/abs/2510.20984)
  learns group-specific lattice generators and companders.
- [Lattice Transform Coding](https://arxiv.org/abs/2403.07320) shows why a
  lattice latent can approach vector rate-distortion behavior that a scalar
  latent cannot.

Those papers principally optimize activation-weighted error or model quality,
not this audit's pooled source-relative MSE.  They therefore motivate the
additive/high-dimensional probes but do not establish this target.  For an
i.i.d. Gaussian source, no code can beat `2^(-2R)`; any 20% improvement here
must come from authenticated non-Gaussian or dependent structure in the Qwen
weights, not merely a better Gaussian code.

## Claim boundary

No early-kill oracle is presented as a universal impossibility theorem.  The
evidence does establish that rate allocation, marginal shape, short-vector
dependence, local contexts, aligned roles, neuron permutations, ordinary
low-rank/tensor factorizations, and SC path search are individually far too
small.  A final success claim requires all of the following in one artifact:

- actual physical rate in `[2.15, 2.5]`;
- independently decoded pooled source-relative MSE satisfying the same-rate
  formula;
- maximum cold per-expert compressed read amplification below `2x`; and
- a manifest binding the container, source hashes, decoder evidence, rate
  ledger, and tamper checks.
