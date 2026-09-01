# Rate-relative research checkpoint: radical gates v9

Date: 2026-09-01
Checkpoint status: **audited search checkpoint; target not achieved**

Machine-readable evidence pins are in
[`results/qwen/rate_relative_research_checkpoint_universal_v9/checkpoint_manifest.json`](../results/qwen/rate_relative_research_checkpoint_universal_v9/checkpoint_manifest.json).

## Fixed objective

The required artifact remains a literal PTQ object satisfying

```text
2.15 <= R <= 2.5 physical bits/weight
F = relative_MSE * 2^(2R) <= 0.8
s = -0.5*log2(F) >= 0.16096404744368115 bpw
maximum cold/page bytes read for one routed expert < 2x
```

Every codeword, table, basis, label, index, header, checksum, alignment byte,
padding byte and reserved capacity bit is charged.  A model- or source-fitted
quantity is not decoder-visible unless the serialized artifact or a fixed
public procedure supplies it.  Independently measured gains are not added;
any composite must rebuild and encode the same decoded residual under one
joint rate ledger.

The architecture is also required to be universal across SwiGLU-MoE models.
Its decoder may use fixed SwiGLU role/shape semantics, bytes in the selected
expert packet, and explicitly transmitted bounded state.  Model identity,
checkpoint ancestry, an external reference model, a router, calibration
activations, or a pretrained model-specific side network is not eligible
decoder state.  Encoder-fitted tables remain eligible only when they are
physically transmitted and their fitting rule is model-agnostic.

The independently decoded expert-affine finite baseline remains
`R=2.5`, relative MSE `0.030902167403153148`, `F=0.9888693569009007`,
`s=0.008074080480766676`, and worst cold-page read
`1.1694444444444445x`.  The strongest honest *ideal* nested structural oracle
remains `F=0.936397620983144`, `s=0.0474034129`; it is not a finite codec and
still lacks `0.11356063457 bpw` of source-specific advantage.  Equivalently,
the missing decoder-visible structure must capture about `14.5662%` under the
frozen favorable screen.

## Audited radical-gate ledger

Rows below are scoped early-kill experiments.  They must not be read as
additive improvements or as replacements for the finite baseline.

| Gate | Strongest honest evidence | Physical rate / cold read | Verdict and boundary |
|---|---|---|---|
| PMG1 tetrad auxiliary | raw/centered capture `-0.04577527/-0.03655885`; delete-expert 3-SE ceiling `0.00155383` | no serialized codec or read ledger | Policy reject and far-short stop on the auxiliary panel; no Gate matrix or pinned validation |
| Free-order SwiGLU predecessor | gross `s=0.01446386`; charged optimistic `F=1.01449256` | permitted rates; `1.00278--1.00543x` | Hard kill even under a relaxed predecessor oracle |
| MALT64 rank-3 tangent | capture `0.09172602`; 3-SE ceiling `0.09185863` vs `0.29724434` required | exactly `2.5 bpw`; `1.01388889x` | Continuous-span kill, not a finite codec |
| SPECTRAFLAG | ideal `F=0.95251520`, `s=0.03509299` | `2.5 bpw`; only an unaudited textual `<1.02x` claim | Early kill; no independent audit, matched Gaussian or polar-Jacobian correction |
| Sparse polar-normal | apparent exact-continuous `F=0.77235840`; eight controls mean `F=0.77239957`; source-specific `s+3SE <=0.00010113` | nominal `2.5 bpw`; producer logical `1x` | Target-looking row is an impossible free continuous-value channel reproduced by controls; no finite win |
| Latent-four meta-codebook | held-out `q=0.52679044/0.52305320`, `F=8.07899/8.02168` | permitted rates; six-expert `1.15833x` at 2.5 | Independently confirmed kill of the latent-four parameterization only |
| Direct-output d8 table | held-out `q=0.10600960/0.10595875`, `F=1.792962/1.792102` | permitted rates; six-expert `1.30x` at 2.5 | Independently confirmed kill without code collapse; ideal residual only |
| QSB v0 | frozen 97% policy exceeded by 892--1,962 bits; no true physical-cap violation | `2.15278/2.30556/2.5`; `1.01389--1.01613x` | Audit blocks the original physical-overflow wording; no distortion result |
| QSB v1 | source-leaking capture `0.733695--0.745511`; derived `F=5.26599--8.14366` | same rate/read triplet | Independently confirmed hard kill; no simulator or controls warranted |
| Decoded affine correction | best favorable transferred `F=0.99878594` | 2.5-bpw envelope; no serialized read field | Numerical kill with an explicitly unsealed pre-execution provenance limitation |
| Same-layer alignment | pooled capture `0.01553490`; favorable `0.01653490` vs `0.14566208` required | no serialized rate/read field | Narrow Up/Down ancestry kill; selected pairs were hashed but not emitted |
| CCQ code-cluster gate | source `q=0.13829420`, `F=2.62965243`; Gaussian `F=2.61318681`; monotone lower bound `F=0.84224257` after ordinal 4 | permitted rates; `1.00278--1.00543x` | Independently confirmed kill of the paper-derived finite cluster/context gate; not an official CCQ encoder |
| KBVQ-IDRE analytic gate | legal role-specific rank 8: `F=1.06894170`, `s=-0.04809159`; even illegal free rank 256 gives `s=0.14688268` | `2.5 bpw` with `0.06518555` side + `2.43481445` residual; `1.075x` | Source-only kill of standalone two-role IDRE; no additive-composite claim |
| RAVEL-6144 v0 | proposed decoded-residual LUT did not run | proposed `0.00462963 bpw` side; `1.18055556x` | Independent source audit blocks launch: wrong raw-SSE projection, hash-then-reopen inputs, and underspecified packet |
| RAVEL-6144 v1 | weighted source-leaking oracle captures only `0.00057309`; oracle `F=1.01765194`; finite FP16 table `F=1.01874211` | `0.00462963 bpw` side; `1.18055556x` | Clean CuPy hard kill; independent result audit passes with the explicit oracle-replay limitation |
| TACTIC-DH384 v2 | conditional dyadic rank-384 residual span; no payload result | exactly `2.5 bpw`; `73/72 = 1.01388889x` | Source-only candidate passes 11 hostile tests and synthetic CPU/CuPy parity; blocked until an authenticated actual `307/128` coarse stream exists, so its planning capture is not evidence |
| Universal lossless wrapper | zstd-19 reduces 8,847,360 bytes to 8,840,025; only `0.00207265 bpw` saved and projected `F=0.98603211` | `R=2.49792735`; approximately one sequential read | Read-only diagnostic kill; xz/gzip/bzip2 are no better and no new codec claim is made |

No row in this table is a finite target-achieving codec.  The only displayed
`F<0.8` is the sparse-polar impossible continuous oracle, and its apparent
advantage is reproduced by matched spatial controls.

## Independent evidence pins

| Evidence | Verifier outcome | Sealed SHA-256 |
|---|---:|---|
| QSB-v0 result audit | PASS, 167 checks | manifest `1a434772a6bf5345465cadd1c9ffb387187ba60249100b4ad4f2dd9c100c7456` |
| QSB-v1 result audit | PASS, 284 checks | manifest `32e12324a1d23e8691bfc5be00df5b384909bee4d7f904f561377977d17fcbc6` |
| Direct-output d8 audit | PASS, 280 checks | manifest `8f8efb1eff6705083c148c02a3176addced24eadb4d43689f6937388a280854e` |
| Decoded-affine audit | PASS_WITH_LIMITATION, 359 checks | manifest `1b010d7dd569289aff20956ac308d66687f25c30b97dc576a9445f313a0b0acb` |
| Same-layer audit | PASS_WITH_LIMITATION, 121 checks | manifest `b8ccb9fb7e21e16dab92a36b4c039d33e8b8d5913771c6f6f787e29bf018b1a2` |
| Latent-four meta audit | PASS_KILL_CONFIRMED_LATENT4_ONLY, 711,480 checks | audit source manifest `5121655f4475df6095536f4295c1ace649d8430dd106a61ee2c2275d974bb9bf` |
| CCQ result audit | PASS_KILL_CONFIRMED, 1,099 checks + 15 tests | manifest `0229f82e0bc9ffcae99e2d67a877ed71e3fd7128917559fef74307552a1c6160` |
| KBVQ-IDRE source gate | PASS source-only | package manifest `c8d644f7366459a0785246545d564c688989dd72c64702dd16cd08f31630ac47` |
| RAVEL-v0 blocker audit | PASS_AUDIT_BLOCKS_PRODUCER_LAUNCH, 131 checks + 7 tests | manifest `2c32fbbca7b4b90bb3d8f769f6f8d85df90eaf8dc16acd08dcc17e51d048cdc4` |
| RAVEL-v1 source and producer result parse | PASS, 89 checks + 10 tests; result PASS, 85 checks | source manifest `ffafd386ab4f3777fb6c9a70fa413f3bdf169658c64607f898a7969d0375c359`; result `ef67ee26246149472b5f3e4dc6f7e869d95c325a354abd7339bc8e4137dc0c47` |
| RAVEL-v1 independent result audit | PASS_WITH_LIMITATION, 212 checks + 9 tests | manifest `e2855085f5e1fe7e20575df660c80eb15a4b3d1b654eccb01ae04bed5510df6a` |
| TACTIC-DH384 v2 source closure | PASS, 11 hostile tests; synthetic-only CuPy parity PASS; no independent audit or payload run | source manifest `f8de593784638cf7719d08ddda7061f4912166021214fb7a2894862a53050662`; synthetic receipt `c91c702a603848abf0b7bfdac0c8e01740bf8cd4d80baa93b0243647fc2c043f` |

The relocated QSB-v0/v1, direct-output, decoded-affine, same-layer and
meta-codebook audits were replayed with native `/usr/bin/python3.12 -B -I` in
the RunPod project root.  Their restored producer closures also pass: QSB-v0
182 checks, QSB-v1 196 checks, direct-output source PASS, same-layer source
PASS, and meta-codebook source PASS.  CCQ was replayed separately with all
1,099 checks and 15 hostile tests.  RAVEL-v0 initially exposed a receipt
canonicalization defect in its own new audit; the seal was repaired and the
subsequent Linux replay passed all 131 checks and seven tests.  The failed
pre-repair replay is not treated as evidence.  RAVEL-v1's independent result
audit likewise had two failed pre-seal attempts; after correcting the receipt
count and ordinal JSON-key ordering, a fresh native RunPod replay passed all
212 checks and nine hostile tests.  Only the final manifest above is evidence.

## Exact replay examples

From the repository root on the RunPod:

```bash
/usr/bin/python3.12 -B -I \
  research/ccq_raw_mse_stage0_v0_independent_result_audit_20260901/verify_audit.py \
  --root "$PWD"

/usr/bin/python3.12 -B -I \
  research/meta_codebook_whole_expert_stage0_v0_result_audit_20260901/verify_audit.py \
  --root "$PWD"

/usr/bin/python3.12 -B -I \
  research/ravel_decoded_residual_lut_stage0_v0_independent_source_audit_20260901/verify_audit.py \
  --audit-dir research/ravel_decoded_residual_lut_stage0_v0_independent_source_audit_20260901
```

The audit READMEs contain the complete source-verifier and hostile-test replay
commands.  Run results belong in sibling `*_runpod_result_20260901`
directories; producer packages remain exact source closures.

## Next search frontier

RAVEL-v1 implemented its entire repair contract—weighted raw-SSE least
squares, held-descriptor authentication, fixed noncyclic semantics, an aligned
versioned packet with an independent parser, finite-FP16/zero-padding checks,
completion written last, and per-matrix sums.  Its corrected oracle is far
short, so enlarging or training this LUT family is not justified.  The result
and its independent narrow audit are retained as a closed negative branch.

TACTIC-DH384 v2 is the next bounded decoder-conditioned cell.  Its frozen
source-free selector maps same-block integer coarse symbols to a rank-384
dyadic frame, leaving 384 target bits per 4,096-value block.  Its exact ledger
is `2.5 bpw` and `73/72` cold-read amplification.  However, the repository has
no authenticated finite `307/128` coarse artifact or canonical integer-symbol
stream.  Consequently the candidate has not read a payload or launched CuPy,
and its favourable transferred `29.7244%` required-capture calculation is only
a launch gate.  Work first has to produce and independently round-trip that
literal universal coarse stream; the existing 2.5-bpw result cannot stand in
for it.

The next architecture search is deliberately crossing outside standard
weight quantization into decoder-synchronised conditional transforms,
invertible lifting/graph seriation, cyclostationary polyphase prediction,
Hankel/finite-rate-of-innovation structure, and universal sequence models.
Checkpoint/reference-delta and initializer-provenance branches are excluded by
the universality contract.  A candidate is promoted only if its
decoder-visible, physically charged oracle can supply at least the missing
`0.11356063457 bpw`, preserves `<2x` cold reads, and has an explicit portability
test across held-out SwiGLU-MoE layers or models.

## Claim boundary

This checkpoint authenticates scoped negative results and preserved source
closures.  It does not claim that the final target is achieved, that the ideal
composite is finite, that missing read ledgers can be inferred, or that an
oracle gain can be transferred to the finite baseline without re-encoding the
same residual.  A success still requires a byte-derived physical rate, a cold
page-union read ledger, causal decode, independent scoring against the pinned
original BF16 weights, matched-control interpretation, and a sealed result
artifact with `F<=0.8`.
