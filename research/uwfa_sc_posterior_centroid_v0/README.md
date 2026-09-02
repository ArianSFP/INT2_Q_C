# UWFA-SC posterior-centroid v0

Date: 2026-09-02

Status: source-only, nonpromoting discovery diagnostic. No model payload,
original BF16 source, completed v9 result, or control source was accessed while
building this package.

## What this tests

This package is the smallest decoder-legal continuous posterior diagnostic for
a completed UWFA-SC v9 literal result. It leaves v8 and v9 byte-for-byte
unchanged. A one-page suffix wrapper, `CAGEPST1`, contains the unchanged
`UWFCV8` object and one cross-fitted binary16 posterior head.

The selected SC branch decisions are **not** treated as scalar quantizer-bin
labels. They are causal polar-decoder decisions. The only scalar coordinate
used by the head is the coordinate-aligned lattice index `k_i in [0,63]`
regenerated after the complete six-level SC decode. The diagnostic hard-fails
unless it can independently obtain all of the following from the literal
object and authenticated decoder sources:

1. every selected decision, public level and base-frequency commitment;
2. the winning unifilar candidate and its pre-decision state trace;
3. the final coordinate-aligned `previous`/lattice-index array for every
   block;
4. the block-to-source group mapping, scale and RHT seed; and
5. the exact full reconstruction digest.

Original BF16 values are fit/score targets only. They are never decoder side
information.

## Frozen law

For coordinate `i` in block `b`, let

```text
q_i = 0.25 * (k_i - 31)
```

be the existing normalized lattice reconstruction. Replay the winning UWFA
over the complete selected-decision stream. For level `l` and pre-decision
state `s`, define

```text
p[b,l,s] = count(b,l,s) / count(b,l,*)
z[b,l,s] = p[b,l,s] - 1/S                 when the level is nonempty
z[b,l,s] = 0                              otherwise.
```

The sole state-aware continuous emission law is

```text
mu_i / scale_b = q_i + a + c*q_i
                 + sum_(l,s) z[b,l,s] * (u[l,s] + v[l,s]*q_i).
```

This is a conditional mean for the continuous coordinate residual, not a
truncated-bin formula. At `S=64`, it has `2 + 12*S = 770` parameters. There is
no 64-entry source-fitted centroid table in v0. The local control retains only
`a,c`. The state-destruction control independently permutes state coordinates
inside every `(block ordinal, level)` before fitting, preserving each
occupancy multiset while destroying consistent state identity.

All features are functions of the literal message, public shape/role, and
fixed decoder code. Layer, expert identity, checkpoint name, tensor name and
external reference weights are forbidden model inputs. Expert ordinals are
used only to derive connected ownership components and route reconstructed
matrices.

## Leakage-proof cross-fitting

Connected components are derived from stream owner sets. For each outer
component `Ck`, every matrix and stream touching `Ck` is held out. With the two
remaining components `Ca,Cb`, each ridge exponent is evaluated in both inner
directions:

```text
fit Ca -> exact original-domain SSE on Cb
fit Cb -> exact original-domain SSE on Ca.
```

The winning exponent is the minimum summed validation SSE, with ascending
exponent as the exact tie break. It is then refit on `Ca union Cb`, serialized
to binary16, independently parsed, applied through the exact inverse RHT and
Up/Down transform, and scored only on `Ck`. Identity, local-only,
state-aware, and state-permuted heads undergo the same nesting.

The fixed ridge grid is:

```text
[-28, -24, -20, -16, -12, -8, -4, 0]
lambda = 2**exponent.
```

The solve uses normalized FP64 sufficient statistics. Every feature,
including the intercept, has a strictly positive standardized ridge penalty.
This makes the centered-state gauge unique.

Three fold heads are three different packets. Their pooled cross-fit score is
a discovery diagnostic, not one deployable codec. A final all-component head
may be refit only after the cross-fit family survives; it must be emitted and
scored as one separate literal packet. Portability still requires a sealed
evaluation on a disjoint SwiGLU-MoE family.

## Source-side score manifest

The runtime accepts a generic `swiglu-bf16-score-panel-v0` manifest. Each row
contains only:

```text
expert_ordinal, role, shape, relative_path, bytes, sha256
```

Roles are exactly `gate`, `up`, and `down`; Gate/Up are `[I,H]`, Down is
stored `[H,I]`. The caller must provide the expected manifest SHA-256
explicitly. The serialized posterior head does not contain that manifest,
source hash, model name, layer or expert identity.

## Physical grammar

`CAGEPST1` is a suffix wrapper:

```text
[unchanged page-aligned UWFCV8 bytes]
[one 4096-byte extension page]
```

The extension page contains the `CAGEPC0` head at its start, zero padding,
and a 192-byte footer at its end. The footer binds inner/head byte lengths,
SHA-256 digests, the complete posterior handoff root, weights, expert count,
fold ordinal and a CRC. The parser slices the inner object before giving it to
the unchanged v8 decoder, so v8's trailing-byte rejection is preserved.

The maximum v0 head is 1,636 bytes including its 96-byte header, so every head
fits in exactly one new 4-KiB page. No second compressed-expert pass is
allowed. The diagnostic executes the exact authenticated v8 causal routed
decoder once per expert through an instrumented view of a literal wrapper,
parses the posterior head from one suffix-page request, and binds every inner
request range and causal reconstruction back to v9 telemetry. Overlapping
parser requests remain charged as requested-with-repetition; they are not
mislabelled as a second pass. The strict read gate includes all three views:
descriptor pages, requested bytes with repetition, and unique requested
bytes.

There is deliberately **no inference-ready routed posterior decoder in v0**.
The implemented scorer reconstructs the full offline coordinate panel; the
routed instrument does not yet accumulate occupancies and apply `CAGEPC0` to
the selected expert inside that same causal session. Therefore the one-page
result is a nonpromoting read projection from an actually executed inner
routed decode plus a literal suffix-page read. Result ledgers explicitly set
`actual_posterior_wrapper_routed_decode_executed=false`. A routed inference or
read-promotion claim is blocked until that decoder exists and reproduces the
offline reconstruction without a second inner read. Scratch/HBM traffic is
reported separately and is not called cold storage traffic.

For fold selection, the inner container byte ledger is allocated by its exact
owner sets and the global extension page is allocated in proportion to source
weights. This ledger is not a literal heldout-only packet and cannot establish
final physical `R` or `F`. Final claims require the literal all-component
wrapper.

## Promotion gates

For every outer component, define

```text
Delta_s = (R0 - R) - 0.5*log2(D/D0)
G_state = Delta_s(state) - max(Delta_s(local), Delta_s(permuted)).
```

The family is killed if any binding/re-encode/reconstruction check fails, any
fold has `Delta_s(state) <= 0` or `G_state <= 0`, binary16 serialization
reverses the gain, rate leaves `[2.15,2.5]`, `F > 0.8`, or maximum routed cold
read is `>=2x`. A source survivor remains nonpromoting until the separately
frozen state permutations, within-group structure destruction, and all eight
matched-Gaussian source pipelines are run and independently replayed.

## Source-only test

```bash
python -I -B \
  research/uwfa_sc_posterior_centroid_v0/test_source_only.py
```

The tests synthesize coordinate-aligned decoded blocks and continuous targets;
they do not open a repository payload or initialize CUDA. They check the
coordinate-alignment fail-closed rule, state-trace semantics, permutation
control, whole-component split, ridge nesting, binary16 head round-trip,
suffix-wrapper canonicality, tamper rejection, exact-v8 authenticated
dataclass loading, retained authenticated sibling execution, and exact
byte/read accounting including overlapping and over-2x request traces.

## Explicit discovery launch

After an independent source audit, pass the published `SOURCE_MANIFEST.json`
SHA-256 explicitly. The output directory must not already exist. The runner
authenticates its own complete source closure before opening the completed v9
publication, and authenticates the complete v9 publication before opening any
BF16 score source. `posterior_core.py` and `result_bridge.py` are compiled from
the retained bytes authenticated by that closure; they are not imported later
from mutable sibling paths.

```bash
python -I -B research/uwfa_sc_posterior_centroid_v0/diagnostic.py \
  --authorization RUN_UWFA_SC_POSTERIOR_CENTROID_V0_DISCOVERY \
  --package-manifest-sha256 <audited-package-manifest-sha256> \
  --v9-result-dir <completed-v9-result-directory> \
  --v8-package research/unifilar_wfa_entropy_census_stage0_v8 \
  --strata-common strata_expert_local_codec/common.py \
  --frozen-auditor strata_v2_klt_mixed_independent_auditor_v1.py \
  --source-manifest <generic-source-panel-manifest.json> \
  --source-manifest-sha256 <authorized-score-manifest-sha256> \
  --rht-device cupy \
  --output-dir <new-output-directory>
```
