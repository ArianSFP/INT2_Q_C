# UWFA-SC v8 independent source-only audit

Date: 2026-09-02

## Verdict

**PASS_SOURCE_FREEZE**

V8 repairs V7's sole remaining source blocker. The outer held-out score is now
one direct difference between the authenticated current payload/framing layout
with zero UWFA-model bytes and the candidate layout with the selected serialized
model exactly once at its aligned placement. There is no equal-model
cancellation or raw model-byte subtraction. V7 gates 1, 2, 4, 5 and 6 remain
intact.

This verdict means the authenticated source is eligible for the repository's
separate manifest/freeze procedure. This audit did not create that authority.
It did not edit the producer, open Qwen/model payloads, run production data or
commit anything.

## Authentication and execution

- Local tree: `research/unifilar_wfa_entropy_census_stage0_v8`.
- Isolated RunPod tree: `/workspace/uwfa_v8_independent_audit_586b127a`.
- Exact inventory: 17 regular files, 633,177 bytes; see
  `SOURCE_INVENTORY.tsv`.
- Independent UTF-8-ordinal source root, computed before Python execution:
  `586b127aa67c88608816a39fbe35888e71d7c00b2928de550420b5e8ae392f18`.
- Local and RunPod sizes/SHA-256s matched exactly; `namei` found no symlink
  ancestor.
- Full POSIX suite:
  `/usr/bin/python3.12 -I -B test_source_only.py`.
- Result: 68 tests in 77.877 seconds; 67 pass, one expected pre-manifest skip,
  zero failures/errors.

## V7 blocker review — PASS

[`stage0_census.py`](../unifilar_wfa_entropy_census_stage0_v8/stage0_census.py)
`_literal_panel_layout_bits` (lines 694-767) keeps authenticated baseline stream
geometry and takes an explicit model byte count. The current comparator passes
zero at lines 788-796. Candidate scoring serializes the selected model and
passes its exact length once at lines 770-785.

Inner selection uses this candidate layout for all 150 cells (lines 1620-1632).
After refitting on development data, the outer fold measures:

- current baseline with zero UWFA model (lines 1635-1637);
- the same baseline payloads with the selected model, as an aligned model-cost
  diagnostic (lines 1638-1640);
- candidate test payloads plus that model exactly once (lines 1641-1644).

The authoritative saving is directly
`literal_baseline_bits - literal_candidate_bits` (lines 1660-1665). No raw
`physical_model_bytes` subtraction remains. Pooled and component gates consume
that exact field (lines 1690-1736).

[`container_codec.py`](../unifilar_wfa_entropy_census_stage0_v8/container_codec.py)
uses one shared integer layout authority (lines 181-300): zero model bytes are
legal for the authenticated current-format comparator; frames are aligned to 64
bytes; the model and following directory are placed at 4-KiB boundaries; regions
and rate-floor padding use the same grammar. The candidate serializer calls
this function and asserts frame, region and total equality (lines 627-725).

The source regression at `test_source_only.py` lines 1335-1375 constructs an
actual serialized candidate comparator and proves its byte length equals the
measure-only fold score. The baseline side is deliberately the authenticated
current payload/framing measurement with no UWFA model, not a decodable UWFA
container requiring a transmitted model.

The boundary regression at lines 688-730 is genuinely discriminatory:

- real two-state serialized model: 1,602 bytes;
- retired raw proxy: 398-byte apparent win;
- exact aligned comparator: 4,096-byte loss;
- four- and eight-state cases verify one- and two-page aligned increments.

Thus the old proxy would select the opposite conclusion from the repaired
physical comparator.

## Unchanged gates

1. **Held/bounded consumption — PASS.** `VerifiedOutputBundle` retains marker,
   directory and member descriptors through use; bounded reads recheck identity
   and digest and never reopen member paths. Path replacement, mutation and cap
   tests pass.
2. **Marker publication — PASS.** Exact marker bytes/digest and `st_nlink=1`
   are checked after descriptor-addressed linking and cleanup; mutation,
   replacement, extra-hardlink, `O_TMPFILE`, `AT_EMPTY_PATH`, `/proc/self/fd`
   and all three directory substitution tests pass.
3. **Telemetry and GPU identity — PASS.** Exact all-150 and representative
   workloads are fieldwise bound; canonical UUID/PCI validation and resealed
   telemetry-forgery rejection pass.
4. **Dependence components — PASS.** Shared streams join owners into disjoint
   connected components; fewer than two holds before fit; promotion requires
   pooled absolute target plus every component positive, with no iid t-test.
5. **Prior regressions — PASS.** Source/control closure, eight controls,
   selectors `0..149`, resource ordering, triplet commitments, physical
   rate/read/symbol-density diagnostics and E250 unequal/shared-tail portability
   remain covered.

## Independent RTX 5090 receipt

One source-free replay was run with transaction id
`43434343434343434343434343434343`. Independent verifier
`independent_audit.py` was authenticated on both hosts with SHA-256
`e8200b5bef35bfae98d1b1015e89555b1e77aac1bc21af01a08b274823cc557d`.
It imports no producer module and verified exact source bytes, result members,
seals, marker identity/sole link, directory root, canonical GPU identity and
ordered selectors through held descriptors.

Receipt identity:

- status: `PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY`;
- payload/public-claim flags: false;
- producer source root:
  `9335f5407aa41eee6e38fe31fcd36b753c77a9f8e4eb2a030d2809c543a4735d`;
- bound preflight:
  `8e33e3a276bfd4f01d8297be506d8b43fe585d262a0c5b37303dc20127124102`;
- receipt SHA-256:
  `c7b8f5243d994a21b3723bcc2beb338cf6e44485f8439251cf144b5ab82a39ae`;
- directory root:
  `eec7c60d4f34096183b0c48bcdb9483e5e18f49bb35b4d1deb0a74e213ad3944`;
- parent commit:
  `8cd26f9ddea03c6a0521edec1377e9082204b23e6485d9210781af11f2ea54f3`;
- parent-marker file SHA-256:
  `7db4b41673ac9ebc7cfdf711d8feb01fb09b33f4210960b3818d60c48c9e20f8`;
- completion seal:
  `2a50a947799c994de48efffebf7c8a9b00b4a71e402c25efc21f8f978deed6b0`;
- final directory device/inode: `66307 / 12215048`;
- marker device/inode/link count: `66307 / 12903102125 / 1`;
- GPU: NVIDIA GeForce RTX 5090,
  `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722`, PCI
  `00000000:16:00.0`;
- all-150: 150 ordered cells, 14.4835724560544 seconds;
- representative: 184,852,206 updates in 28.2900240009185 seconds,
  projected 2,178.16429813704 seconds; phase updates sum exactly.

Authenticated receipt copies are retained beside this report. Their file
SHA-256s are:

- `COMPLETE.authenticated-copy.json`:
  `372fcc5be0481779cc6476c0c1a116d807be9f8e2a61d21bc299d26f27c71704`;
- `RUN_STATE.authenticated-copy.json`:
  `42533915e22af1daab644c4a11c345db6a46a30c896dd07fdcc049659f760f1b`;
- GPU receipt and marker: as listed above.

## Exact limitations

- Source-only PASS; no statement about Qwen entropy, MSE, compression gain or
  runtime performance.
- No production payload was opened, statted, hashed or enumerated.
- No manifest, freeze, authorization, publication or git commit was created.
- The baseline fold comparator is the authenticated current payload/framing
  measurement with zero UWFA model; standalone decode applies to the candidate
  container, which includes the transmitted model.

Within that stated source-only scope, no blocker remains.
