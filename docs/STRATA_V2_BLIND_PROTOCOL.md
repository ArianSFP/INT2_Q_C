# STRATA-XKLT-SC v2 second blind-panel protocol

> **Current publication state:** complete. The selection, codec freeze,
> pre-access validation, source finalization, allocation lock, one-shot intent,
> physical artifact, independent source-domain audit, and tamper receipt are
> published at their canonical paths. The primary gate passed; the result is
> reported in [STRATA_V2_RESULTS.md](STRATA_V2_RESULTS.md).

## Why a second panel exists

The first frozen blind STRATA run completed operationally and independently
decoded, but missed its scientific threshold:

| Historical v1 quantity | Audited value |
|---|---:|
| Route-inclusive physical bits | 60,867,264 |
| Source weights | 28,311,552 |
| Physical rate | 2.14990912543403 bpw |
| Source energy, FP64 | 16,659.395474367368 |
| SSE, FP64 | 860.6248940275871 |
| Pooled relative MSE | 0.05166003144302383 |
| Gaussian reference | 0.050765774772264724 |
| Ratio to reference | 1.017615345668824 |
| Verdict | **Fail: 1.7615% above reference** |

V1 is now development evidence. V2 cannot retroactively turn it into a pass,
and the two panels may not be pooled after seeing v1's outcome.

## Deterministic second-panel selection

The only selection seed is the raw SHA-256 of the immutable v1 independent
audit summary:

```text
6aa4d8538389179cdbdb2edaf332d707eef39aa4ef7cd395f8e82d755ca1bb37
```

For every eligible `(stratum, layer, expert)` candidate:

```text
score = SHA256(seed_bytes || uint8(stratum) || uint8(layer) || uint8(expert)).
```

There is no salt, timestamp, retry counter, or free-form domain. The
lexicographically smallest eligible score wins in each of six eight-layer
strata. Eligibility excludes every layer coordinate and every expert
coordinate appearing in a conservative 39-pair prior-access ledger, not just
previously used pairs.

The precommitted triplets are:

| Stratum | Layer range | Layer | Expert | Matrices |
|---:|---:|---:|---:|---|
| 0 | 0–7 | 5 | 18 | gate, up, down |
| 1 | 8–15 | 12 | 7 | gate, up, down |
| 2 | 16–23 | 18 | 20 | gate, up, down |
| 3 | 24–31 | 28 | 83 | gate, up, down |
| 4 | 32–39 | 36 | 76 | gate, up, down |
| 5 | 40–47 | 45 | 41 | gate, up, down |

This yields eighteen matrices, 108 original `2^18` blocks, and 28,311,552
BF16 weights. Index and safetensors-header metadata established all shapes,
shards, and byte ranges before source access; selected payload hashes were
null at proposal time.

## Pre-access sealed proposal

| Artifact | SHA-256 |
|---|---|
| Selection proposal file | `528250d8c6bac52dfdf64958d7f4929a115ff68d907a47880cab85d532aade14` |
| Selection internal seal | `cd8cb70ca7509d2ddd4899df8a7047b7b8f47d381b637e2eb497db9ecd4eb9f8` |
| Literal 144-byte route | `94feb3564fe0c3eddfc745703f1f6001b5ae316e7146209e6b45323cdf81697c` |
| Route audit | `a95b17ff26027b6a76ad42c04b2b1e655fb80307d168a795f7c7e6c5305de22c` |
| Unopened snapshot audit | `0d0a7de5ecca5f6ca914841dcbad028275c2c36ca31dffcf8cd37c0fc975ebe3` |
| Preserved v1 failure audit | `5ebe6fd5efbc10162a49a84083e99ae0123daa1680fd54b47a12d79f99369ea3` |

The proposal validator rederived selection and route bytes, checked the
internal and file seals, verified layer/expert disjointness, required every
future source hash to remain null, and refused a workspace containing a v2
codec freeze, materializer, selected source file, unblinded directory, or full
checkpoint shard.

## Codec freeze before materialization

Only after development and negative tests were complete did the canonical
builder create the v2 freeze. The freeze binds:

- exact selection and literal route bytes;
- architecture and byte-format constants;
- exact `<=2.15 bpw` ledger and strict `<2^-4.3` MSE gate;
- codec, emitter, runner, independent auditor, and tamper-test hashes;
- unit/contract/synthetic tests and development evidence;
- the Python interpreter executable;
- complete installed NumPy, SciPy, CuPy, and cuda-pathfinder distribution
  trees, including wheel RECORD verification; and
- CUDA runtime/driver, compute capability, and RTX 5090 device identity.

The canonical freeze was then validated in a fresh process while the selected
payloads remained absent:

| Artifact | File SHA-256 | Internal seal |
|---|---|---|
| Codec freeze | `1cd70bfc3147663b15039d0f949e5e4b999bf1c6981d33d21dbc4f1b7d7ebda7` | `7191f2813ec70be7793613f82722dc82c253c6eed72f9098796c88aad2173779` |
| Pre-access validation receipt | `9b2472b8ffefd96f0ed36919b0d431818229985820daf82ca1e9c8e5455e8c2d` | `f022028caf1303587f8c5713768135a3f33b1f28fa0388f6aa0abb7cfe8a2ef7` |

The receipt records that all expected future outputs and the unblinded
directory were absent, selected proposal hashes were null, full shards were
absent, and the materializer source file itself did not yet exist.

## Authorized source finalization

The fail-closed materializer was introduced only after freeze validation. It
accepts no tensor, URL, range, or output-path override. It requires the exact
authorization phrase frozen in the codec lock, fetches only the eighteen
sealed inclusive byte ranges, and requires:

- HTTP status 206;
- exact request and `Content-Range` values;
- identity content encoding;
- exact byte count;
- one create-only output per matrix; and
- full-matrix and six nested block SHA-256 values.

The finalized source lock covers all 18 matrices, 108 blocks, 56,623,104
source bytes, and 28,311,552 BF16 values. It is bound to the exact selection,
codec-freeze, and validation-receipt seals.

```text
source-lock file SHA-256: bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23
source-lock internal seal: 5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b
```

Those hashes establish source identity, not codec quality.

## Pre-encoding source-derived lock

The emitter reads each source file once into an authenticated immutable byte
snapshot, recreates the KLT staging, labels, block permutation, energies, DP
profiles, and seeds, then writes all staging files and control assets. Before
any encoder subprocess exists, it seals:

- `preencoding_manifest.json`;
- `allocation.lock.json`;
- `header.bin`, `route.bin`, `labels_3bit.bin`, and `profiles.bin`; and
- fourteen complete BF16 staging blocks.

The one-shot intent binds those files, every runtime artifact, the exact
environment receipt, the planned fourteen invocations, and the rule
`retry_resume_or_adaptive_rate_change_allowed = false`.

Canonical pre-encoding hashes are:

| Artifact | File SHA-256 | Internal seal, where applicable |
|---|---|---|
| Pre-encoding manifest | `5cdced62fe7ccba39e2e313bf40a79820e285fbb86323424a8da697161dd6539` | n/a |
| Allocation lock | `65efad51f92047236217ff1e90da2fc5cbac567de99ac8ff845a1533e5b1632a` | `6679dddd181a2199f1a1b57d360949369cf7fb174ef8007615650abfae2a4141` |
| One-shot intent | `0878e9844d6e8249e51e2e4a6586c436a73b9671c5cf3e11ae7cff500c2bf299` | n/a |

Allocator-projected MSE and encoder-side staged metrics are not the blind
result. They must not be substituted for the independent original-domain
score.

## One-shot execution rule

The frozen runner:

1. verifies the blind runtime freeze before opening finalized source payload;
2. completes and seals every source-derived decision;
3. writes the intent before the first encoder invocation;
4. invokes each of fourteen stateless block encoders exactly once;
5. refuses resume, an existing output directory, or a development flag in
   blind mode;
6. rehashes runtime files, package trees, CUDA identity, staging, manifest,
   and allocation after encoding;
7. packs once into the fixed physical reservoir; and
8. writes either a completed summary or an explicit failure record.

If any encoder fails or the byte-padded payloads exceed the reservoir, the
run is a negative result. The protocol forbids deleting its lock, changing a
profile, or retrying a convenient seed.

## Independent audit

The independent auditor does not import the encoder, its probability trace,
its selected decisions, or a reliability table. It must:

- parse the 7,608,729-byte container and exact rate ledger;
- regenerate KLT coefficients from Q15 codes;
- rederive labels, long-block staging, scales, profiles, SC seeds, and RHT
  seeds directly from authenticated sources and control bytes;
- rerun the exact allocation DP;
- construct all BEC frozen sets procedurally in unsigned Q31;
- causal-decode every decision and canonically arithmetic-reencode every
  payload byte;
- invert RHT, stream order, and scaled-orthogonal KLT;
- score every original BF16 value once in FP64; and
- verify complete selection → source → freeze → allocation → intent → summary
  lineage plus the independently measured frozen runtime.

The auditor separates `audit_execution_passed` from
`primary_claim_gate.passed`. A correctly decoded artifact can still be a
scientific failure due to rate, MSE, incomplete lineage, or non-blind mode.

After the completed audit, the tamper harness must demonstrate rejection of
ten independently resealed mutations, including source identity, threshold,
profile, runtime binding, header coefficient, padding, and FP16 scale changes.

## Current gate table

| Gate | State in this repository revision |
|---|---|
| Historical v1 result preserved as failure | complete |
| Second-panel selection sealed before access | complete |
| Codec and exact rate ledger frozen | complete |
| Freeze independently validated before access | complete |
| Exact sources materialized and hash-finalized | complete; compact source lock published |
| Allocation and one-shot intent sealed | complete; both locks published |
| Fourteen one-shot encodes | complete: 14 invocations, 0 retries/resumes |
| Fixed container emitted and reservoir fits | complete: 7,608,729 bytes, 7,096-byte zero tail |
| Independent decode/re-encode | complete: 14/14 byte-exact |
| Independent original-domain FP64 score | complete: `0.04985939119332436` |
| Ten lineage tamper tests | complete: 10/10 rejected |
| Final blind pass/fail verdict | **PASS** |

## Publication rule

The precommitted rule required a negative outcome to be published unchanged if
any primary condition failed. Because every condition passed, the supported
positive statement is deliberately narrow:

> The frozen STRATA-XKLT-SC v2 codec beat the `2^-4.3` Gaussian reference on
> its second precommitted 18-matrix Qwen expert panel at an exact
> route-inclusive physical rate no greater than 2.15 bpw.

That statement must be accompanied by the exact panel/container hashes,
independent source energy, SSE, pooled MSE, physical integer ledger, decode
status, and claim limitations. It must not be generalized to a full model or
used as a perplexity result. Those quantities and limitations are recorded in
the final result card.
