# SwiGLU neuron-permutation predictive-codec oracle

## Decision

**Hard kill.** On the pinned STRATA-v2 Qwen3-30B-A3B panel, this family is
more than an order of magnitude short of the required structural advantage.
The strongest deliberately illegal pair oracle removes `0.9738465528%` of
source energy, equivalent to only `0.0070592471 bpw`. The goal requires
`-0.5 log2(0.8) = 0.1609640474 bpw`, so the upper bound supplies just
`4.3856%` of the required gain.

The corresponding best legal one-to-one match removes `0.7409223445%`
(`0.0053645231 bpw`). Its favorable information-theoretic map and FP16
three-scalar side cost is `0.0091391669 bpw`, making its net rate gain
negative even before quantizing the reference or predictor.

## What was tested

The exact six expert triplets / eighteen matrices from the checkpoint plan
were authenticated and loaded as BF16. For each expert, each intermediate
neuron is represented by the geometrically legal joint vector

```text
[gate row (2048), up row (2048), down column (2048)].
```

All 30 directed expert pairs were tested, including every available
cross-layer/cross-expert direction. A permutation always moves the same 768
channels in gate, up, and down, preserving SwiGLU neuron semantics. Tested
mappings include identity, combined-norm sorting, gate-norm sorting,
within-neuron signature sorting, signature-space Hungarian assignment, exact
joint-cosine Hungarian assignment, and exact role-wise Hungarian assignment.

The strongest legal oracle chooses the permutation that maximizes captured
energy and fits three independent real-valued least-squares coefficients for
each matched channel. The stronger many-to-one ceiling lets every target
channel reuse its individually best reference channel. That is not a
permutation and cannot be a legal neuron reindexing, so failure there is a
strict early-kill signal.

| Optimistic construction | Maximum removed energy | Equivalent gain |
|---|---:|---:|
| Identity, three exact scalars/channel | `0.0729649732%` | `0.0005265231 bpw` |
| Signature Hungarian, three exact scalars/channel | `0.0796948317%` | `0.0005751059 bpw` |
| Exact joint-scalar Hungarian | `0.4947011422%` | `0.0035773704 bpw` |
| Exact role-wise three-scalar Hungarian | `0.7409223445%` | `0.0053645231 bpw` |
| Illegal many-to-one role-wise reuse | **`0.9738465528%`** | **`0.0070592471 bpw`** |

The best legal direct-reference star over the entire panel removes
`0.3508373993%` of panel energy (`0.0025352067 bpw`). Even the impossible
reuse star removes only `0.4052676372%` (`0.0029293279 bpw`). The best causal
tree improves the legal result insignificantly to `0.3581447218%`, while
requiring a dependency depth of four and therefore violating a two-stream
cold-read budget.

## Physical side cost and read bandwidth

An arbitrary permutation of 768 channels costs at least
`ceil(log2(768!)) = 6,260 bits`; a simple packed map costs
`768 * 10 = 7,680 bits`. Three FP16 coefficients per channel add `36,864`
bits. Even using the impossible-to-beat enumerative map lower bound, the side
cost is therefore

```text
(6,260 + 36,864) / (3 * 768 * 2,048)
    = 0.0091391669 bpw per predicted expert triplet.
```

For a cold random expert read, a direct-reference codec must read one root
stream plus the target residual. Under the favorable Gaussian scaling model,

```text
read amplification = 2 + (side_bpw - oracle_gain_bpw) / R.
```

Every tested legal pair has `side_bpw > oracle_gain_bpw`. At `R=2.5`, the
best case is `2.0015099x` and the worst is `2.0029051x`; none satisfies the
strict `<2x` condition. Keeping the reference permanently decoded would move
the accounting near `1x`, but that is a cache-residency assumption rather than
a per-expert cold-read result, and references span different model layers.

## Why this is not the old exact-coordinate test

The earlier structural-redundancy probe tested source-channel reuse on a few
development pairs and already reported about `0.3%` capture. This audit
extends that idea specifically to the current frozen 18-matrix panel, enforces
the shared gate/up/down neuron geometry, evaluates exact one-to-one Hungarian
permutations, compares source-local norm/signature canonicalizations, builds
causal full-panel star/tree layouts, and includes a physical map/scalar and
read-amplification ledger. Its result is consistent with, and stronger for
this checkpoint than, the earlier negative evidence.

## Reproduction and evidence

Run on the source-holding machine with GPU visibility disabled:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
  python neuron_permutation_oracle.py \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --output neuron_permutation_oracle_result.json
```

- Reproducer: `neuron_permutation_oracle.py`
- Result: `neuron_permutation_oracle_result.json`
- Script SHA-256: `5d2b91c1fa42b4f8793eaeb69acd3a1b1dd2fb65f8f98197f96d01e470c705dc`
- Result SHA-256: `3bdff037e24fdd853e569419df8cc769c53d4be04f90c043b35403adbd66bfbd`
- Plan SHA-256: `99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`
- Independently recomputed source energy: `16192.894508855932`
- Directed pairs: `30/30`
- Runtime: `9.9472 s`, NumPy/SciPy CPU backend only

The JSON records all eighteen expected and observed source SHA-256 hashes,
all pair metrics, exact strongest legal mappings, mapping hashes, graph
parents/depths, side-cost calculations, execution versions, and the script
hash. The source energy agrees with the frozen STRATA-v2 independent audit to
floating-point display precision (`16192.89450885593`).
