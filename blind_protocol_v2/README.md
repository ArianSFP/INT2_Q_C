# Second untouched Qwen panel proposal

> Historical protocol document: statements about unopened payloads describe
> the state when this proposal was sealed. The one-shot run has since completed
> and passed. The freeze-bound base encoder is withheld from redistribution for
> unresolved upstream licensing; its path and SHA-256 remain in the freeze.

This directory contains a metadata-only, sealed proposal for a second blind
panel on the pinned `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

It is intentionally **not** a codec freeze and **does not** authorize source
materialization or encoding. No selected BF16 tensor payload has been opened.

## Why a second panel is needed

The first frozen blind run was operationally valid but missed the distortion
gate. `audit_v1_result.py` independently rescored the already-opened v1 sources
and canonical reconstruction with CuPy FP64 reductions:

| Quantity | Re-audited value |
|---|---:|
| Route-inclusive physical rate | 2.14990912543403 bpw |
| Physical bundle | 60,867,264 bits / 28,311,552 weights |
| Source energy | 16,659.395474367368 |
| SSE | 860.6248940275871 |
| Pooled relative MSE | 0.05166003144302383 |
| Gaussian reference at 2.15 bpw | 0.050765774772264724 |
| Ratio to reference | 1.017615345668824 |
| SSE reduction needed for a strict pass | 1.7310416695% |

Integrity and exact physical rate passed; MSE failed. The defensible v1 result
is therefore a clean negative result on its precommitted 18-matrix panel. A v2
result cannot retroactively turn v1 into a pass, and the two panels must not be
pooled after observing their outcomes.

The v1 error was not uniform. Down projections pooled to 0.0507559183, just
below the reference, while gate and up projections pooled to 0.0524674141 and
0.0517955189. Across expert triplets, relative MSE ranged from 0.0430481460
at L44:E111 to 0.0573610792 at L16:E108. This is useful development evidence,
but it also makes a genuinely untouched confirmatory panel essential.

## Deterministic selection

The only selection seed is the raw SHA-256 of the immutable v1 independent
audit summary:

`6aa4d8538389179cdbdb2edaf332d707eef39aa4ef7cd395f8e82d755ca1bb37`

For each candidate the score is:

```text
SHA256(seed_bytes || uint8(stratum) || uint8(layer) || uint8(expert))
```

There is no salt, timestamp, retry number, or free-form domain string. The
lexicographically smallest eligible score wins independently in each of six
eight-layer strata.

Eligibility is deliberately stricter than pair-only disjointness. The audit
ledger contains 39 previously opened or conservatively reserved `(layer,
expert)` pairs. It excludes every layer index and every expert index appearing
in that ledger. The selected panel therefore uses wholly new layer coordinates
and wholly new expert coordinates, not merely new combinations.

- Excluded layers: 0, 3, 11, 15, 16, 22, 25, 31, 32, 44, 46, 47
- Excluded expert indices: 0, 5, 8, 13, 15, 16, 24, 31, 32, 40, 48, 56, 57,
  63, 64, 72, 75, 80, 87, 88, 96, 104, 108, 111, 112, 120, 121, 125, 127

L46:E13 is only a conservative reservation from an older source-free protocol;
the ledger does not falsely assert that its payload was fetched.

The selected expert triplets are:

| Stratum | Eligible pairs | Layer | Expert | Winning score |
|---:|---:|---:|---:|---|
| 0 (0–7) | 594 | 5 | 18 | `0095dcca41a1fa34edb2a6f242d82ea98c27b0147f3ecead2de56a322b6c1eb4` |
| 1 (8–15) | 594 | 12 | 7 | `00356c52a31262d500851d2f835ddd3148411c6a51334981777186f079f3e1c5` |
| 2 (16–23) | 594 | 18 | 20 | `00242dc0cb3a802e232b1f0fa9bc848aeca85b4c9a4b7ede8a5dc285a4de9b8d` |
| 3 (24–31) | 594 | 28 | 83 | `00231cbfa2018562d38490c942fa46121447cda2436c93dda56c720899e3bf21` |
| 4 (32–39) | 693 | 36 | 76 | `0032f648df9073bd39a04a4701233bfe9f4db0b27e0f28048741a1f5b51a2964` |
| 5 (40–47) | 495 | 45 | 41 | `002a324197f639fa8756987e27e1dc8bcb3ef41edfd7d61e2a2d2d629b8a06ad` |

Gate, up, and down projections travel together for every selected expert. The
panel contains 18 matrices, 108 blocks of 262,144 weights, and 28,311,552 BF16
weights total.

## Exact metadata-validated ranges

These byte ranges are recorded for future use but remain unopened. They are
absolute inclusive HTTP ranges in the pinned safetensors shard.

| Ordinal | Tensor | Shard | Inclusive range |
|---:|---|---|---:|
| 0 | `model.layers.5.mlp.experts.18.gate_proj.weight` | 00002 | 3115499824–3118645551 |
| 1 | `model.layers.5.mlp.experts.18.up_proj.weight` | 00002 | 3118645552–3121791279 |
| 2 | `model.layers.5.mlp.experts.18.down_proj.weight` | 00002 | 3112354096–3115499823 |
| 3 | `model.layers.12.mlp.experts.7.gate_proj.weight` | 00004 | 2848105616–2851251343 |
| 4 | `model.layers.12.mlp.experts.7.up_proj.weight` | 00004 | 2851251344–2854397071 |
| 5 | `model.layers.12.mlp.experts.7.down_proj.weight` | 00004 | 2844959888–2848105615 |
| 6 | `model.layers.18.mlp.experts.20.gate_proj.weight` | 00006 | 3184707048–3187852775 |
| 7 | `model.layers.18.mlp.experts.20.up_proj.weight` | 00006 | 3187852776–3190998503 |
| 8 | `model.layers.18.mlp.experts.20.down_proj.weight` | 00006 | 3181561320–3184707047 |
| 9 | `model.layers.28.mlp.experts.83.gate_proj.weight` | 00010 | 613577192–616722919 |
| 10 | `model.layers.28.mlp.experts.83.up_proj.weight` | 00010 | 616722920–619868647 |
| 11 | `model.layers.28.mlp.experts.83.down_proj.weight` | 00010 | 610431464–613577191 |
| 12 | `model.layers.36.mlp.experts.76.gate_proj.weight` | 00012 | 2460660528–2463806255 |
| 13 | `model.layers.36.mlp.experts.76.up_proj.weight` | 00012 | 2463806256–2466951983 |
| 14 | `model.layers.36.mlp.experts.76.down_proj.weight` | 00012 | 2457514800–2460660527 |
| 15 | `model.layers.45.mlp.experts.41.gate_proj.weight` | 00015 | 1333957096–1337102823 |
| 16 | `model.layers.45.mlp.experts.41.up_proj.weight` | 00015 | 1337102824–1340248551 |
| 17 | `model.layers.45.mlp.experts.41.down_proj.weight` | 00015 | 1330811368–1333957095 |

The selector validated all 18,432 expert-projection entries in the checkpoint
population from the cached index and all 16 JSON headers. Every selected tensor
is BF16 with the expected shape and exact 3,145,728-byte span. Candidate
availability and offsets came only from index/header metadata. The selector has
no network client and does not open tensor payload files.

## Sealed artifacts

| Artifact | SHA-256 |
|---|---|
| `selection.proposal.lock.json` | `528250d8c6bac52dfdf64958d7f4929a115ff68d907a47880cab85d532aade14` |
| Selection internal content seal | `cd8cb70ca7509d2ddd4899df8a7047b7b8f47d381b637e2eb497db9ecd4eb9f8` |
| `route_table.proposal.bin` | `94feb3564fe0c3eddfc745703f1f6001b5ae316e7146209e6b45323cdf81697c` |
| `route_table.proposal.audit.json` | `a95b17ff26027b6a76ad42c04b2b1e655fb80307d168a795f7c7e6c5305de22c` |
| `unopened_snapshot.audit.json` | `0d0a7de5ecca5f6ca914841dcbad028275c2c36ca31dffcf8cd37c0fc975ebe3` |
| `v1_failure_independent_audit.json` | `5ebe6fd5efbc10162a49a84083e99ae0123daa1680fd54b47a12d79f99369ea3` |

The proposed literal route is 144 bytes (`18 × 8`) in `>HHBBH` records and
would cost 1,152 physical bits if adopted. Its bytes are deterministically
derived from the selection; there is no route sidecar or hidden permutation.

## Safe verification on the RunPod

The following operations make no v2 network request and read no v2 payload:

```bash
/workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/blind_protocol_v2/audit_v1_result.py \
  --workspace /workspace/INT2__compression \
  --output /workspace/INT2__compression/blind_protocol_v2/v1_failure_independent_audit.json

/workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/blind_protocol_v2/prepare_selection_proposal.py \
  --workspace /workspace/INT2__compression \
  --metadata-cache /workspace/INT2__compression/qwen_weight_cache \
  --output-dir /workspace/INT2__compression/blind_protocol_v2

/workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/blind_protocol_v2/validate_proposal.py \
  --workspace /workspace/INT2__compression \
  --proposal-dir /workspace/INT2__compression/blind_protocol_v2
```

`validate_proposal.py` independently checks file and content seals, derives the
route bytes again, verifies all source hashes remain null, checks layer/expert
disjointness, and refuses a state containing a v2 codec freeze, unblinded
directory, materializer, selected BF16 payload, or full checkpoint shard.

## Required protocol before any unblind

1. Treat v1 as development evidence from now on. Architecture work may use it,
   but no v1 metric is held out any longer.
2. Finish and independently audit the complete codec, allocator, serializer,
   decoder, rate ledger, seeds, and one-shot runner without v2 source access.
3. Create a new codec freeze that binds the exact selection file hash/internal
   seal above, the literal route hash, every executable/profile hash, and the
   exact route-inclusive `<= 2.15` physical-rate inequality.
4. Validate that freeze in a fresh process before a v2 materializer exists.
   The freeze must refuse non-null proposal source hashes and must record that
   `blind_protocol_v2/unblinded` did not exist.
5. Only after the freeze, introduce a fail-closed materializer requiring an
   explicit authorization phrase. It may fetch exactly the 18 sealed ranges,
   must require HTTP 206 with exact `Content-Range`, and must create a separate
   source lock bound to both selection and codec-freeze hashes.
6. Seal allocation before encoding; encode each nonzero block once; pack once;
   do not retry, resume, or tune after observing v2 data.
7. Independently unpack the runner-created physical container without invoking
   the packer, decode without importing encoder decisions/probabilities, and
   score every original BF16 value exactly once in FP64.
8. Report integrity, exact physical rate, and MSE as separate gates. Publish a
   negative outcome unchanged if any primary gate fails.

If v2 passes, the narrow claim is that the frozen codec beat the `2^-4.3`
Gaussian reference on this second precommitted 18-matrix Qwen panel at an exact
route-inclusive physical rate no greater than 2.15 bpw. It is not a universal
rate-distortion theorem, a full-checkpoint result, or a perplexity claim.
