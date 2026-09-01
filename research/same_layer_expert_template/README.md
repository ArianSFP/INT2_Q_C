# Same-layer expert-template source-free gate

## Outcome

**Early kill; no new Qwen payload and no GPU job.** The strongest exact,
free-side train-span oracle supplies only `s=0.00000924228 bpw`, versus the
nested remaining requirement `0.11356 bpw`. A separate bounded neuron-alignment
stress test reaches only `0.00179009 bpw` even when it is allowed to fit and
score the same full columns. The preregistered early-stop rule therefore blocks
an all-128-expert fetch.

This directory was created specifically to avoid duplicating two prior
experiments:

- `../neural_flow_oracle/shared_expert_basis_oracle.json` already tests the
  layer mean and leave-one-expert-out matrix PCA on 16 authenticated layer-15
  experts.
- `../../../agent_rd_structure_diag_cross_expert_result.json` already tests
  identity, fixed-reference Hungarian, and two-pass barycenter Hungarian
  neuron alignment with disjoint fit/evaluation columns.

The new question was whether train-only *clustered* prototypes, optionally
after legal held-out neuron alignment, could justify extending that cohort to
all 128 experts. The answer for the bounded family is no.

## Unaligned prototype dominance proof

For a held-out matrix `x`, every train-only direct mean, medoid, cluster mean,
or scalar-gained prototype is in the linear span `S` of the 15 training expert
matrices. Orthogonal projection onto all of `S` minimizes squared residual over
that span:

```text
||x - P_S x||^2 <= ||x - t||^2  for every candidate t in S.
```

Consequently the prior exact rank-15 leave-one-out projection is a favorable
upper bound for every such prototype, including oracle target-dependent
prototype selection:

| Source | Residual `q=F` | `s=-0.5 log2(F)` |
|---|---:|---:|
| Qwen exact train-span oracle | 0.999987187556 | 0.000009242284 |
| Identically fitted iid-Gaussian control | 0.999990463257 | 0.000006879339 |
| Qwen minus control | — | 0.000002362945 |

The Gaussian control is analytic and exact: an independent held-out vector in
`d=1,572,864` dimensions has expected captured fraction `r/d` in an independent
rank-`r` training span. The direct-mean control is `q=1+1/15=1.066666667`; Qwen
is slightly worse at `1.066793197`.

The Qwen rank-15 bound supplies only `0.00814%` of the nested required `s`.
There is therefore no reason to run literal prototype clustering: it cannot
beat the complete-span projection from which its prototypes are constructed.

## Bounded neuron-alignment evidence

Neuron permutation is the only genuinely distinct escape from the span proof.
The authenticated prior study fit permutations on columns `j mod 4 in {0,1}`
and evaluated opportunity on disjoint columns `{2,3}`. Its exact
role-specific FP64 KLT result is compared against an identically fitted,
same-total-rate covariance-diagonal Gaussian control.

| Alignment row | Gain | `F` | Equivalent `s` | Nested need supplied |
|---|---:|---:|---:|---:|
| Disjoint-column reference alignment | 0.009279% | 0.999907205 | 0.0000669403 | 0.05895% |
| Outcome-leaky full-column stress row | 0.247852% | 0.997521479 | 0.001790095 | 1.576% |

The second row is deliberately more favorable than a legal evaluation. It is
still 63.44 times short of the nested requirement before storing a prototype,
permutation, gain, mapping, or residual.

This is a plausibility stop for the named identity/reference/barycenter
column-alignment family, not a mathematical converse for every possible
permutation.

## Physical/read ledger

The most favorable materializable row uses one prototype per fold. Each
crossfit bank is charged as an exact whole-byte 2.15-bpw expert triplet while
its quantization error is optimistically ignored.

| Component | Physical charge |
|---|---:|
| One template triplet | 1,268,121 bytes |
| Two banks amortized over 128 experts | 0.0335937341 bpw |
| One enumerative `768!` permutation | 6,260 bits/expert = 0.00132666694 bpw |
| Three FP16 gains | 48 bits/expert = 0.000010172526 bpw |
| Shared header | 512 bits |
| Total side charge | 0.0349314213 bpw |
| Residual budget under 2.15 bpw | 2.1150685787 bpw |
| Cold read amplification | **1.984374534x** |
| Hot cached-template amplification | 0.984375007x |

Thus the candidate satisfies the `<2x` bandwidth constraint, but has no
information gain. Applying the side ledger to even the outcome-leaky stress
row gives negative `s=-0.0331413 bpw` (`F>1`). More prototypes increase the
amortized bank charge; cluster IDs also become nonzero.

## Why the all-128 run was not launched

The frozen protocol specifies parity expert folds, train-only prototypes,
disjoint alignment columns, a matched Gaussian control, and complete physical
accounting. It also requires stopping before payload access when an optimistic
bound cannot supply `s >= 0.11356`. Both the strict unaligned bound and the much
more favorable aligned stress row fail by orders of magnitude. Fetching roughly
1.2 GB of new expert payload and running CuPy would therefore violate the stop
rule rather than strengthen the conclusion.

No pinned production matrix, excluded payload, model shard, network endpoint,
or GPU was opened for this gate.

## Claim boundary

The result kills unaligned train-only prototypes/clusters and the bounded
identity/reference/barycenter column-alignment family on authenticated
layer-15 Up/Down evidence. It does **not** rule out arbitrary combinatorial or
nonlinear functional alignment. `gate_proj` was not independently measured;
if Up/Down supplied no gain, Gate alone would need `s>=0.34068 bpw`
(`F<=0.623577`) merely to cover the nested triplet requirement.

## Verification

From the repository root:

```text
python research/same_layer_expert_template/verify_source_free_gate.py
```

The verifier checks byte counts and SHA-256 identities of all three permitted
source-free inputs, follows the exact JSON paths, recomputes every `F`, `s`,
Gaussian-control, template, permutation, and read-amplification value, confirms
all access counters are zero, and enforces the early-kill inequalities.
