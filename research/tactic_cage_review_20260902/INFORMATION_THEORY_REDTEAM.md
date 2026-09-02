# TACTIC-CAGE information-theory red team

Date: 2026-09-02

Verdict: the architecture is coherent, but no CAGE performance claim is yet
earned.

| Mechanism | Verdict | Exact interpretation |
|---|---|---|
| Qwen UWFA/CTW recoding | Partial | Valid lossless recoding; the source-free fixture calibrates detection only. No successful Qwen score existed at review time. |
| Coarse-programmed graph/lifting | Partial | A legitimate conditional quantizer. `G=Phi(C)` costs no descriptor but contains no information beyond `C`; only joint conditional geometry can help. |
| Posterior centroid | Mathematical pass; empirical partial | `E[X|M]` is MSE-optimal for a fixed message partition. It requires cross-fitting and a fully serialized model. Local adverse corrections do not close a non-local posterior. |
| 384-bit adaptive tree | Partial | A valid `2^384`-leaf conditional VQ. Adaptivity changes geometry and search, not communicated information. |
| `h(C)` transform selection | Partial | Legal in joint search. There is no free selector; coarse candidate freedom or distortion changes. A `log2(M)` penalty is a heuristic, not a universal theorem. |
| 384-bit syndrome | Block pending source law | A coset is computational structure; the decoder still has at most `2^384` reconstructions per coarse word. |
| Bits-back | Block now | It can approach a latent model's marginal codelength, not beat it or make a fitted state free. Reservoir, posterior gap, termination and random access are physical. |
| TACTIC-DH384 | Block | The independently decoded `307/128`-bpw coarse object does not exist. Layout arithmetic cannot replace it. |
| Universal SwiGLU-MoE | Block as a performance claim | One parser/architecture can be universal, but `F<=0.8` cannot be guaranteed for every source; an iid Gaussian source is a counterexample. Current C4 target eligibility is also N18-divisibility limited. |
| `<2x` traffic | Layout pass; operational block | One planned frame pass is `73/72`; a second is `2.01111x`. No emitted composite/read trace exists, and cold storage pages do not account for HBM graph/BP traffic. |

Bits-back references support only the narrower implementation claim above:
BB-ANS, arXiv:1901.04866, and Bit-Swap, arXiv:1905.06845.

## Quantitative kill gates

### Same-reconstruction UWFA

Require the literal container to be no larger than 8,302,592 bytes:

```text
required saving                    544,768 bytes
required saving                    0.153935185185185 bpw
selected SC symbols                126,627,266
required charged saving            0.03441710571244585 bit/selected symbol
```

### Coarse-residual capture

Once the actual coarse distortion `D_c` exists, a 2.5-bpw fine mechanism must
capture at least:

```text
c_req = 1 - 0.025 / D_c
```

For the planning-only `D_c=0.0355742423`, this is `29.7244%` of coarse error,
versus `9.375%` for an isotropic rank-384 real projection.  Kill a graph/frame
when an optimistic arbitrary-real continuous projection has an upper bound
below `c_req`.  During exact evaluation, stop once accumulated irreducible SSE
exceeds `0.025 * source_energy`.

### Rate-specific target

For every literal candidate:

```text
D_target(R) = 0.8 * 2^(-2R)
```

Rate and distortion must come from the same nested packet/fit.

### Adaptive tree

Use a continuous relaxation of all descendants as a lower bound.  Prune a
prefix only when even its optimistic remaining correction cannot reach
`D_target(R)`.  A finite beam miss without such a bound is not a family kill.

### Syndrome and bits-back

Do not implement a syndrome backend unless its held-out conditional-RD oracle
beats the best direct 384-bit conditional tree after identical model cost:

```text
<0.03 bpw       hard kill
>0.11356 bpw    eligible for one nested composite
>=0.15289 bpw   sufficient in principle for the standalone ideal gap
```

Bits-back must beat explicit latent coding after the posterior gap, startup
reservoir, termination and padding.  A 4 KiB reservoir per expert alone costs
`0.00694444` bpw and one cold page for an N18 expert.

## Required experiment order

1. Finish the literal Qwen UWFA early gate.
2. Build and independently decode the actual lower-rate coarse stream.
3. Run continuous DH384, graph and Krylov residual upper bounds.
4. Only for a surviving conditional geometry, test the cross-fitted posterior
   centroid and adaptive tree.
5. Keep syndrome and bits-back as gated backends, not presumed sources of
   gain.
