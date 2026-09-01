# FOSP-ARX v1: charged free-order SwiGLU path gate

## Decision boundary

The attractive **zero-bit permutation** version is an immediate metric-
compatibility kill.  It is function-preserving, but it is not eligible for the
frozen compression score, which is independently decoded MSE against the
original BF16 arrays in their original coordinates.

The only eligible form in this package transmits the permutation's factoradic
rank inside the same expert frame.  This costs `ceil(log2(768!)) = 6,260`
information bits, physically `783` bytes / `6,264` bits, or
`0.0013275146484375 bpw`.  It still needs only one expert frame read.  The
charged architecture remains a narrow, genuinely untested opportunity, so a
source-only CuPy early-kill protocol is frozen here; no Qwen payload has been
opened and no execution is authorized.

## Exact architecture and why zero bits fail

Write a Qwen expert in the common neuron orientation:

```text
G = gate_proj          in R^(768 x 2048)
U = up_proj            in R^(768 x 2048)
D = down_proj.T        in R^(768 x 2048)
```

For a permutation matrix `P`, deploy

```text
G' = P G,       U' = P U,       down_proj' = down_proj P^T.
```

Elementwise SiLU and multiplication commute with `P`, so this changes no
expert function.  An encoder can choose a path through the 768 joint
`[Gate, Up, Down.T]` neurons and encode each neuron from its predecessor.

That symmetry does **not** make the original labels decoder-visible.  If the
encoder emits only a canonical/reordered payload, a deterministic decoder can
recover one orbit representative but not which of the `768!` labelings was the
original tensor.  Sorting the decoded weights again gives the canonical order,
not canonical-to-original labels.  Hiding those labels in quantizer/codeword
choices still consumes rate.  Therefore the eligible decoder must:

1. read and unrank the inline 783-byte factoradic value;
2. sequentially decode one anchor and 767 joint path residuals; and
3. scatter Gate/Up rows and Down columns back through the inverse permutation
   before the independent original-coordinate MSE comparison.

The factoradic field is side information, even though it is inline rather
than a separately fetched stream.

## What prior permutation work did—and did not—test

This opening is not already measured, but its zero-bit interpretation is also
not a loophole in the prior accounting.

| Prior package | What it tests | Difference from FOSP-ARX |
|---|---|---|
| `neuron_permutation_oracle` | Aligns each target expert to a different reference expert with exact Hungarian Gate/Up/Down mappings and coefficients | Fetches/reference-predicts another expert, charges a map, and never optimizes adjacency among neurons of one expert |
| `permutation_aligned_expert_template` | One auxiliary Up/Down target/reference pair | Omits Gate; its independent audit found the Hungarian objective was not a raw-MSE upper bound; still cross-expert rather than an intra-expert path |
| `permutation_quotient_initializer_anchor_v1` | Aligns a generated initializer anchor to a source expert | Explicitly allocates 783 bytes per neuron permutation and restores source labels; no adjacent residual codec |
| `invariant_manifold_oracle` | Quotients the exact scaling/permutation action before a joint polar model | Also charges 783 bytes to restore original coordinates; it fits a dense polar manifold, not an encoder-selected path |

The closest negative context is the invariant-manifold result: even a very
favorable coupled polar oracle was worse than its matched Gaussian controls.
That makes FOSP survival unlikely but does not numerically contain a
source-chosen within-expert adjacency graph.

Frozen predecessor identities used for this assessment:

```text
5d2b91c1fa42b4f8793eaeb69acd3a1b1dd2fb65f8f98197f96d01e470c705dc  research/neuron_permutation_oracle.py
3bdff037e24fdd853e569419df8cc769c53d4be04f90c043b35403adbd66bfbd  research/neuron_permutation_oracle_result.json
dc9cc00dfbfb9e378291bb0abf67ff9d5c21d62a0f6c1f3c85df1b947d3dca00  research/permutation_aligned_expert_template/permutation_pair_screen.py
ba22a5ac76a6cc697f63899787ab85396a5b00dc2d764299473ecb59e3a52a52  research/permutation_aligned_expert_template/pair_result.json
d375777d76f4b89e92164ea2e36fc0bcb47fe66dacd5e68d68599fcdf0526ed2  research/structural_reference_audit/permutation_aligned_expert_template_audit.json
9329eb9fa1c41a45ff75b55575c12549f5420990d774250e947bac8910b47f8a  research/invariant_manifold_oracle/gauge_coupled_polar_oracle.py
2ca913ae4ce13ac23d7d3b1b9f2867ed6512dc337cca1a4bc385151829131614  research/invariant_manifold_oracle/result.json
2f596940e637935e547145ca344739700585df3ce5495c0e4929824b744444c4  research/permutation_quotient_initializer_anchor_v1/finite_contract.py
```

## Strongest cheap necessary-condition gate

The protocol first grants each expert **three exact, independent, dense,
free** 768-dimensional KLTs—one each for Gate, Up, and canonical Down.  This
is strictly more flexible at second order than one common permutation and a
banded same-role predictor.  Exact continuous reverse-waterfilling is scored
at `2.15`, `2.30`, and `2.50 bpw`.

Every Gaussian control is passed through the identical eigensolver,
waterfill, and rate selection.  It is not merely globally moment-matched.  For
each expert and each neuron separately it preserves:

- all three role means over the 2,048 model coordinates;
- the exact centered `3 x 3` Gate/Up/Down Gram; and therefore
- every row energy, cross-role covariance, and heteroskedastic scale.

Only the centered orientation is randomized.  Eight fixed controls and a
delete-one-expert jackknife give the optimistic statistic

```text
s_Qwen - mean(s_control) + 3*sqrt(SE_control^2 + SE_jackknife^2).
```

If that remains below `0.16096404744368115 bpw`, the linear
adjacency/covariance family is stopped.  This is a second-order opportunity
kill, not a universal converse for nonlinear higher-order codes.

Only a dense-envelope survivor may form all `768*767` directed pairs.  For a
target joint neuron `Y` and predecessor `X`, both `3 x 2048`, the strong pair
oracle fits the nine-coefficient regression

```text
A* = (Y X^T) (X X^T)^-1
capture = tr((Y X^T) (X X^T)^-1 (X Y^T)).
```

It reports both:

- an illegal relaxation in which every target independently chooses its best
  predecessor, allowing reuse, cycles, and no anchor; and
- an achievable one-to-one path obtained from a maximum-weight non-self cycle
  cover, weakest-edge cuts, and deterministic path concatenation.

Nine coefficients per selected edge are the entire per-edge model; there are
no coordinate-wise free parameters.  Exact coefficients are favorable oracle
arithmetic only.  Every proposed physical bridge explicitly charges all
`767*9` or `767*3` coefficients.

## Exact rate and read ledger

One expert has `4,718,592` weights.  The eligible frame contains a 64-byte
header, the 783-byte factoradic permutation, coefficient indices/values, and
the residual payload.  At a requested cap `R`, its size is
`floor(4,718,592*R/8)` bytes; side fields reduce the payload reservoir rather
than expanding the cap.

| Coefficient bridge | Coefficient bits | Coefficient bpw | Total header+map+coefficient bpw |
|---|---:|---:|---:|
| 3 x FP16 / edge | 36,816 | 0.0078023275 | 0.0092383491 |
| 9 x FP16 / edge | 110,448 | 0.0234069824 | 0.0248430040 |
| 3 x fixed 4-bit / edge | 9,208 | 0.0019514296 | 0.0033874512 |
| 9 x fixed 4-bit / edge | 27,616 | 0.0058525933 | 0.0072886150 |

The fixed 4-bit alphabet is decoder-resident and procedural.  A learned table
would need an additional charge.

| Requested cap | Frame bytes | Actual bpw | Logical byte read | Pessimistic cold 4-KiB read |
|---:|---:|---:|---:|---:|
| 2.15 | 1,268,121 | 2.1499989827 | 1.0000x | 1.0045224391x |
| 2.30 | 1,356,595 | 2.2999996609 | 1.0000x | 1.0054349308x |
| 2.50 | 1,474,560 | 2.5000000000 | 1.0000x | 1.0027777778x |

The page ledger pessimistically rereads one shared manifest page.  Decoder
scratch retains one previous BF16 joint neuron (`12,288` bytes) but scratch is
not compressed-object read traffic.

## Source firewall and execution status

`source_bindings.json` names two already-authenticated, non-pinned auxiliary
Qwen triplets (six matrices) and all three roles.  The runner accepts only a
workspace root and refuses any alternate manifest or individual source path.
It reads and hashes each fixed file through one non-following descriptor and
decodes the same bytes it hashed.

This is still **source-only**:

- no Qwen matrix was numerically opened while preparing it;
- the pinned panel is not an accepted input;
- the package authorizes no auxiliary run;
- an independent source audit and source-free CuPy calibration are mandatory
  before any source access; and
- even a source-oracle survivor cannot authorize a finite or pinned run.

The two bound triplets were prior confirmation sources for another branch.
That history is a limitation: they are acceptable as a cheap discovery
screen, not fresh confirmation evidence for FOSP.

## Reproduction after a future authorization

Source-only tests do not import CuPy:

```bash
python -B test_source_only.py
python -B verify_package.py
```

Only after a separately retained independent PASS receipt may a source host
invoke the runner.  The sentinel is deliberately insufficient on its own and
is not an authorization claim:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B \
  free_order_oracle.py \
  --workspace-root /workspace/INT2__compression \
  --output /new/disjoint/result.json \
  --authorization-sentinel INDEPENDENT_SOURCE_AUDIT_PASS_REQUIRED
```

The output is create-new and outside the frozen package.  A source result must
be independently audited before it can inform any next gate.

