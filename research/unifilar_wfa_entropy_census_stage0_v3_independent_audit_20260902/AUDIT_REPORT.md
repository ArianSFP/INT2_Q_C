# Independent hostile source audit — UWFA-SC v3 post-implementation boundary

Date: 2026-09-02  
Producer tree: `research/unifilar_wfa_entropy_census_stage0_v3`  
Verdict: **BLOCK — do not create a source manifest, freeze, result, or payload launch yet**

This is a source-free verdict. It says nothing about whether Qwen contains a
useful long-range dependency. No Qwen/model artifact, current codec payload,
extracted decision stream, matched-Gaussian payload, or numeric result was
opened by this audit.

## Exact authority boundary

The producer had no `SOURCE_MANIFEST.json`, as required at the
post-implementation boundary. Before importing any producer module, the audit
independently enumerated all 16 regular files, rejected symlink ancestry, read
each through a no-follow held descriptor, checked its exact size and SHA-256,
and computed this canonical inventory root:

```text
a1fc85ffdfaa5e7fde25deea98b33d186c915868a8a546333a7fefb64fa9b035
```

The root is SHA-256 over case-insensitively name-sorted records
`name NUL bytes NUL sha256 LF`. The exact member sizes and digests are hard-coded
in both independent launchers. The inspected inventory was:

```text
INDEPENDENT_BOOTSTRAP_ABI.md  README.md             container_codec.py
cupy_backend.py              design_lock.json       dispatcher_contract.py
fixture_long_memory.py       fixture_portability.py protocol.py
result_envelope.py           stage0_census.py       strata_sc_adapter.py
test_source_only.py          universal_adapter.py   uwfa_common.py
verify_source.py
```

The independent tools are not part of the producer tree:

```text
hostile_audit.py  sha256=87698f3c84c51ef1162c156b2ae3f0031fe4cac4910ebade108b66e0aa23f08b
gpu_replay.py     sha256=fec3881fceac9a9a0444106a4a6c38b5229b367e9bcae2959d6b6f8ffbaef035
```

All 16 files and every documented ABI were inspected. The exact machine-readable
receipts are `HOSTILE_RESULTS.json` and `GPU_REPLAY_RESULTS.json` in this audit
directory.

## Independently reproduced mechanics

These parts passed on the authenticated snapshot.

### Producer source suite

Command:

```text
cd /workspace/uwfa_v3_independent_audit_a1fc85ff
/usr/bin/python3.12 -I -B test_source_only.py
```

Result: 38 tests executed in 58.421 seconds; 37 passed and the one expected
pre-freeze manifest test skipped. There was no unexpected failure.

The suite successfully covered:

- exact binary arithmetic and Q0.16 model canonicality;
- frozen 150-cell topology/reset bank;
- fixed 32-byte owner sets at the 1/8/32/128/256 boundaries;
- bounded header rejection before semantic parsing;
- nonzero padding, trailing bytes, model, directory, frame, and contribution
  mutations;
- canonical full rebuild;
- unselected corrupt-frame non-read behavior;
- owner-aware dual-denominator attribution, including the hostile case where
  `total/E` reports `1.8823529x` but the exact owner-local page ratio is `3.2x`;
- forged and cross-backend GPU handles failing before a device call;
- completion-last behavior for ordinary injected member/post-completion faults;
- direct producer and in-tree dispatcher self-authorization rejection.

### New 250-expert unequal-shape fixture

The independent hostile launcher built a new universal fixture with:

- 250 experts;
- 12 distinct `(hidden, intermediate)` shape pairs;
- 751 streams and 251 owner regions;
- all Gate/Up/Down scalar intervals covered exactly;
- one shared Gate tail owned by experts 0 and 249 with unequal contributions
  of one and two scalars;
- descriptor-backed fresh routed decode for every expert.

The literal container SHA-256 was
`4b29ecd60f637be958d6abd3db181341cac5ae001bb951829780e06929cc7b59`.
Canonical rebuild passed, every routed payload decoded and canonically
re-encoded, and the concatenated routed FP64 reconstruction matched the bound
full-reconstruction digest. This is a portability result, not a useful-rate or
cold-amplification result: the deliberately tiny matrices make metadata/page
ratios unrepresentative.

### Fresh RTX 5090 source-free replay

Commands:

```text
/workspace/int2-cupy-venv/bin/python -I -B \
  /workspace/uwfa_v3_independent_audit_tools_a1fc85ff/gpu_replay.py \
  /workspace/uwfa_v3_independent_audit_a1fc85ff

nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,pci.bus_id \
  --format=csv,noheader
```

Independent host observation:

```text
NVIDIA GeForce RTX 5090, 32607 MiB, 580.126.09,
GPU-c06e0fe0-9836-2f98-8f10-0514d085f722, 00000000:16:00.0
```

All-150 replay:

```text
status             PASS_ALL_150_CPU_CUPY_EXACT_REPEATED
cells              150
streams            4
symbols/bank       7702
elapsed            14.606124197 s
cell receipt SHA   c5e33aced64dfb5f6fc2d3ea9101ceaee9e126c125bde0a47fdcc4a1b7e53cc9
```

Representative all-150 outer fold:

```text
status                  PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD
measured updates        184,588,779
measured time           28.060213060 s
measured throughput     6,578,309.958 updates/s
conservative throughput 3,289,154.979 updates/s
13-pipeline projection  4,306.726528 s
budget                   21,600 s
kernels                  304
H2D                      9,628,560 bytes
  payload                7,151,808
  root descriptors             240
  subset descriptors        24,688
  launch descriptors        24,688
  model tables           2,422,272
  kernel scalars             4,864
D2H                     19,380,720 bytes
```

The measured phase includes pack, complete candidate search, refit, score,
literal serialize/parse/decode/re-encode/rebuild, and excludes only the stated
0.002562-second warm-up. The selected source-free proxy cell was suffix,
`chi=2`, reset 4096. This independently reproduces the CPU/CuPy mechanics and
feasibility only. It is a direct-copy pre-freeze replay, not the required final
fresh replay from an exact public Git commit.

## Freeze blockers

### B1 — the scientific fold grammar is not total on legal universal panels

`panel_geometry` accepts any unique semantic `(layer, expert)` pairs. Both
`projected_updates` and `nested_holdout` include a development stream only when
every owner differs from the outer identity in **both** layer and expert
coordinate:

```text
owner_layer != outer_layer AND owner_expert != outer_expert
```

That stronger two-axis holdout is deliberate and useful when the panel has a
sufficient layer-by-expert cross-product. It is nevertheless undefined for a
legal single-layer SwiGLU-MoE panel. The independent counterexample used six
unique identities `(0,0)..(0,5)` and three private role streams per expert.
Exact-pair exclusion would leave 15 development streams per fold; the frozen
rule left zero and raised:

```text
ValueError: nested fold geometry
```

This does not invalidate the codec grammar or a sufficiently rich Qwen panel.
It blocks the advertised universal scientific protocol. Before freeze, either:

1. declare and validate a narrower panel precondition before any fit, with a
   proof that the intended Qwen and cross-model panels satisfy it; or
2. freeze a total fold ladder, such as exact-identity outer holdout for every
   legal panel plus the stricter layer-and-expert-coordinate exclusion as an
   additional diagnostic when nonempty.

Silently accepting the panel in geometry validation and failing later in the
runtime estimator is not fail-closed universality.

### B2 — source geometry and source-free preflight receipts are not gated

The source path receives `BoundEvidence.source_panel_sha256`, but never compares
it with the recomputed `protocol.geometry_sha256(panel)`. The geometry is first
computed after nested fitting and final packing. A source-free interface probe
injected:

```text
actual geometry = aa...aa
bound geometry  = bb...bb
gpu preflight status = FORGED_PRETEND_PASS
```

`source_phase` returned normally and copied the forged preflight record into
its result. The mock replaced downstream numeric work only to isolate this
interface gate; it did not assert a payload result.

Related static findings:

- `gpu_preflight` status, exact 150-cell count, equality receipt, environment,
  and source root are never validated by `source_phase`;
- the representative benchmark receipt is not an input to, or gate of, the
  source phase;
- the score receipt's `original_source_panel_sha256` and
  `independent_decoder_source_sha256` are checked only as syntactically valid
  SHA-256 strings, not against a held expected source/decoder identity.

The independent dispatcher may own some of these checks, but it does not exist
at this boundary. Freeze must wait until either the producer validates exact
typed receipts before the first fit, or an independently pinned dispatcher does
so and its hostile audit proves that no call into `source_phase` is possible
otherwise. At minimum, recomputed source geometry must equal the value embedded
in the literal container.

### B3 — output publication follows a symlink ancestor

`CompletionLastOutput` canonicalizes path text, calls `os.mkdir(path)`, then
opens only the new leaf with `O_NOFOLLOW`. It neither walks nor retains the
output parent descriptor. The independent Linux counterexample created:

```text
alias -> real/
CompletionLastOutput(alias/result)
```

The transaction completed successfully and wrote `real/result/COMPLETE.json`.
This contradicts the no-follow/retained-descriptor trust model. Ordinary
post-member and post-completion fault tests still pass, but they do not cover
ancestor substitution.

The fix is a retained, authenticated parent directory supplied by the external
bootstrap, followed by descriptor-relative exclusive `mkdirat/openat`, retained
child identity, member writes through that descriptor, completion last, and
directory fsync. Add symlink-ancestor and parent-rename/substitution tests.

### B4 — declared CUDA bounds permit allocation beyond the frozen budgets

The backend declares:

```text
MAX_PACKED_SYMBOLS = 17,179,869,184
minimum packed payload device bytes = 4*N = 68,719,476,736
frozen stage VRAM budget             = 30,064,771,072
observed RTX 5090 total VRAM         = 33,668,857,856
```

`MAX_HOST_BYTES` and `MAX_VRAM_BYTES` are reported by `projected_updates` but
are not used in a rejection predicate. `pack_streams` can concatenate host
payloads and issue CuPy copies after only checking `MAX_PACKED_SYMBOLS`.
Therefore not every accepted bound is safe before dependent host/device
allocation.

Before freeze, derive one exact host/device byte budget from stream lengths,
descriptors, worst candidate outputs, model tables, and a stated margin; reject
it before any blob concatenation or CuPy call. The gate should also use measured
free VRAM and current process RSS, not merely emit constants in a receipt.

### B5 — the CUDA receipt does not bind device 0 to a UUID/PCI identity

The backend receipt reports device id, name, compute capability, runtime,
driver API version, and memory. It contains neither a device UUID nor a PCI bus
id. The independent `nvidia-smi` observation above proves that this particular
run used an RTX 5090, but that mapping is outside the producer receipt and
cannot be joined cryptographically to its telemetry.

Add UUID and PCI bus id to `environment_receipt`, compare them against an
independently obtained NVML/driver record, and reject a mismatch. The exact
all-150 mechanics already pass; this blocker concerns evidence identity, not
CUDA arithmetic.

### B6 — control equality mixes structural geometry with source-dependent lengths

`panel_geometry` includes `baseline_payload_bytes` and
`baseline_logical_bits`. `controls_phase` then requires the complete control
geometry digest to equal the source geometry digest. In the independent valid
panel, changing only one `baseline_payload_bytes` field changed the digest:

```text
source  85287f432bb3c336b5188cc0227eb0113cd712b0af923945dc4452dd9976bb36
altered 0e4bef2a61ad1efbb6f75507044a189f72161c41ffad819747ea0792a44c5f7a
```

An independently re-encoded moment-matched Gaussian artifact will ordinarily
have different source-dependent arithmetic lengths even when identities,
shapes, routes, symbol counts, roles, ownership, and pipeline are identical.
As written, such a legitimate control is rejected before any fit.

Freeze a `structural_geometry_sha256` for cross-source equality that excludes
source-dependent payload/logical lengths. Separately bind the complete full
geometry of each source/control to its own artifact and score receipt. If the
intended control generator instead guarantees identical baseline logical
lengths, that unusual invariant must be documented and independently replayed
before this blocker can be cleared.

## Smaller correction required before freezing the diagnostic ABI

The posterior handoff contains no posterior arrays, MMSE result, extracted
source result, or free reconstruction data; that part passes. It binds literal
container, model, context state, semantic routes, reconstruction identity, and
per-stream digests.

However, the STRATA plugin's `source_digest` is a length-prefixed digest of
`(selected bits, levels, base frequencies)`, while the handoff publishes that
value under the field name `decoded_symbol_bits_sha256`. The independent probe
confirmed the combined digest differs from the actual bit-only SHA-256. Rename
the field to state its triplet semantics, or add a separately reproduced
bit-only commitment. Do not freeze an ambiguous downstream ABI.

## Gates that are not blockers in this snapshot

- The arithmetic decoder uses only the current public SC context and prior
  decoded bits. State resets before the first lookup at each reset boundary and
  transitions only after the current bit. No future selected bit enters a
  frequency lookup.
- The STRATA routed decoder regenerates SC levels/base frequencies from literal
  inherited metadata and the already-decoded prefix. It does not consume a
  selected-bit, level, or base-frequency side file.
- Authoritative cold evidence requires an `AuthenticatedDescriptorSource` and
  a routed decoder session. Memory routing is explicitly diagnostic and cannot
  set `passes_cold_read_below_2x`.
- Owner attribution uses exact fractions and the worse total/nonpadding
  denominator. The `total/E` shortcut is not used.
- Directory/region/frame owner copies, complete scalar intervals, rate rational
  `43/20`, terminal bit padding, region padding distribution, and canonical
  re-encode/rebuild are checked.
- Fresh launch descriptors are rebuilt from backend-private host tuples;
  forged/cross-backend/mutated public descriptors fail before CuPy calls in the
  tested cases.

## Required re-review boundary

Do not patch this audited tree in place and claim the receipt still applies.
Build a new source snapshot, then repeat:

1. exact 16-member pre-import authentication with a new root;
2. the complete producer source-only suite;
3. all hostile probes above, including legal same-layer identities, source
   geometry/preflight mismatch, symlink/rename output attacks, memory-bound
   rejection before allocation, control structural/full geometry separation,
   and posterior commitment semantics;
4. fresh all-150 and representative RTX 5090 replay with UUID/PCI mapping;
5. only then independent source review, manifest/freeze, public-commit replay,
   and external dispatcher audit.

No producer file was modified by this audit. No manifest, freeze record, result,
payload artifact, commit, or Qwen claim was emitted.
