# ORBIT/MPS source-model research checkpoint

Date: 2026-09-01
Status: mathematical assessment complete; source-free nonlocal mechanism
verified; all production payload cells blocked before Qwen access.

## Scope

This checkpoint evaluates the proposed ORBIT–MERA–MPS–RCC architecture under
the universal SwiGLU-MoE contract.  Qwen is an evaluation family only.  Model
ancestry, external base checkpoints, layer/expert identity keys, routers and
activations are forbidden decoder information.

The current independently decoded finite baseline remains:

```text
R = 2.5 bpw
D = 0.030902167403153148
F = D * 2^(2R) = 0.9888693569009007
```

Keeping that reconstruction unchanged would meet `F <= 0.8` if a lossless
source model saved at least `0.15288996696291447` physical bits per source
weight.  This equals `4,328,552.2499` bits over the six-expert panel.

## Research verdict

The proposal contains one high-value empirical hypothesis: a small causal
long-memory model may predict legal quantizer symbols better on held-out
SwiGLU-MoE weights than on the complete matched-Gaussian pipeline.

The other components are conditional:

- MERA/lifting is promoted only after a fixed-transform entropy survivor and
  must optimize held-out operational codelength plus source-domain MSE.
- RCC is a stochastic transmission backend.  It does not create source
  structure, supplies expected rather than automatically fixed-cap rate, and
  has impractical generic search at large KL.
- Spatially coupled LDGM is a possible finite search backend, not a source of
  below-Gaussian performance.
- Orbit/Gray–Wyner remains a lower-priority multi-expert oracle.  Alignment can
  expose common information but cannot create it, and shared bytes enter every
  routed expert's cold read.
- FIBER/type statistics have no standalone entropy rebate because for a
  deterministic statistic `b=f(q)`, `H(q)=H(b)+H(q|b)`.

The full derivation, literature boundary, exact ledgers and promotion criteria
are in `ORBIT_MERA_RCC_RESEARCH_ASSESSMENT_2026-09-01.md`.

## Verified source-free mechanism

`research/nonlocal_wfa_global_state_synthetic_v0` implements a sparse
symbol-conditioned unifilar WFA and a canonical arithmetic codec.  Its fixture
contains 26 iid bits plus six separated parity checks per 32-bit block.  Every
suffix through depth 25 remains at one population bit per symbol.

RunPod result:

```text
structured logical rate             0.8168277956 bit/symbol
independently refit iid control      1.0000672979 bit/symbol
detected nonlocal difference         0.1832395022 bit/symbol
eight-stream model-charged saving    0.1675415039 bit/symbol
sparse model                         3,456 bytes / one 4 KiB page
```

The independent replay regenerated both sources and decoded every arithmetic
symbol.  It passed 12/12 tests on the provided RunPod.  This proves only that
the implementation detects a genuine suffix-invisible dependency; it is not
Qwen evidence.

Bindings:

- source-manifest SHA-256:
  `529ddcc549f2ea76878c4848a13fd30656e6e92eaa3b8d027d5526927b50c820`
- result SHA-256:
  `9359dfbcee026a9ea02b425725d0c5e8c2b5289cd12912a4532993ddf765ad8a`

## Production-cell audit outcomes

No production cell has payload authority and none opened Qwen, the current
codec artifact, extracted decision arrays or Gaussian controls.

### Dense HMM v0: BLOCK

Path: `research/tied_mps_entropy_census_stage0_v0`
Independent audit manifest:
`020c754f8eaf9bddb8a1ac10ee88ed44d477f21c3ea061befcd439b5712d061d`

The audit found manifest races, full-panel hyperparameter leakage into
holdout, fold identity acting as a model selector, float-defined decoded
probabilities, a continuous-probability hard-kill inconsistent with the
physical Q0.16 coder, incomplete Gaussian replay, mismatched HMM timing,
reset leakage, symlink/bootstrap defects and infeasible dense runtime.

### Exact-unifilar SC census v1: BLOCK

Path: `research/unifilar_wfa_entropy_census_stage0_v1`
Source manifest:
`1dbea65550d879c3cc6ca81974223d251d669c15f5af17fa9681800cf03cf9ff`
Independent audit manifest:
`df7b78f97c798a9ab0893ca17d4661874278cd21b89ba6418dcfee681fc64366`

The core passed 75,600 transition cases, 2,046 exhaustive arithmetic cases,
all 150 CPU/CuPy cells and linear-scaling tests.  Launch is nevertheless
blocked by source TOCTOU, symlink ancestors, an artifact-size field unbound to
the artifact, final verification that ignored the serialized model, absence
of one literal complete container, and unenforced Gaussian geometry.

### Raw Lloyd-4 label-copula census v0: BLOCK

Path: `research/label_copula_census_stage0_v0`
Source manifest:
`e1bc2873f204b1db5fefa666d0daf6ddebae38bd3f1add3ce45f3bc0538aae14`
Independent audit manifest:
`b8384c70534dbc062425648e1794ed20c2afaf01fdf5b07b43f6e7735640f803`

Orientation, Gaussian-Lloyd labeling, all 240 state laws, Q0.16
serialization, arithmetic decoding and rate/read ledgers passed.  Launch is
blocked because writes were possible after completion, symlinked packages
were accepted, a one-layer bootstrap could masquerade as a confidence bound,
control provenance was not bound to the full pipeline, and reusable expert
slots were not validated.  This stream is diagnostic and would still require
a literal nested lossy re-encode after any survivor.

## Capacity and runtime cautions

A nonnegative bond-`chi` factorization carries at most `log2(chi)` bits of
mutual information across one cut; a Born-amplitude model has a roughly
`2*log2(chi)` ceiling.  At `chi=64`, these are six and about twelve bits.  The
standalone target is approximately `313.119` saved bits per 2,048 weights.
A small state may be reused and therefore can still yield extensive savings,
but one parity constraint saves only `1/2048 = 0.00048828125 bpw`.

Dense causal contraction also costs `O(K*chi^2)` per decoded symbol.  A useful
MoE codec needs sparse/low-rank or fused decoding in addition to a small model
byte count.  The verified unifilar cell uses one integer state update and one
probability lookup per symbol.

## Next authorized work

The immediate next step is not RCC, MERA or a Qwen launch.  It is one
consolidated source repair that:

1. imports only authenticated immutable bytes or retained file descriptors;
2. emits and independently decodes one literal complete packet;
3. binds artifact size, every model byte and Gaussian geometry/provenance;
4. makes completion irrevocably write-final and rejects every symlink
   ancestor;
5. requires multiple independent whole test layers for uncertainty claims;
6. re-runs the independent source audit on a new manifest.

Only an audit pass authorizes the CuPy Qwen census.  A production miss closes
the frozen finite-state cell, not arbitrary MPS/Born/MERA structure.
