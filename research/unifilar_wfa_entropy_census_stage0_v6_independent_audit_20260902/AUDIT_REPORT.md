# UWFA-SC v6 independent source-only audit

Date: 2026-09-02

## Verdict

**BLOCK_SOURCE_FREEZE**

The v6 publication mechanics repair the previously identified staging/final
directory substitution windows, and the authenticated source suite and
source-free RTX 5090 replay completed. They do not establish that v6 is safe to
freeze or consume. Five source-level gaps remain: authenticated result bytes
cannot be safely handed to a consumer, marker content is not revalidated after
publication, candidate/held-out selection is not charged by the literal
container geometry it claims to optimize, GPU telemetry is not semantically
bound to the claimed workload, and the held-out confidence calculation treats
dependent folds as independent.

This is a source-only verdict. No Qwen/model payload was opened, no production
entropy result was run, and no Qwen/WFA gain or compression claim follows from
this audit.

## Scope and independence

I did not build v6. I did not edit the producer, create a producer manifest or
freeze, open a Qwen/model payload, or commit. The only new files are this audit,
its independent hostile harness, exact audit inventory/evidence, and byte-for-
byte copies of the already authenticated source-free GPU receipt records.

The local source and RunPod replay copy were authenticated before execution as
17 regular files, 572,546 bytes, with independent inventory root:

`b14bf19aa8965f0ab22ec26db43cddd63e0c5f3c4d996edeed45e512e516cca2`

See `SOURCE_INVENTORY.tsv` and `TEST_EVIDENCE.md` for exact members, commands,
receipt hashes, inodes, roots and observed results.

## What passed

### Publication protocol exercised by the requested hostile cases

The independent harness confirmed all of the following on the RunPod's actual
Linux filesystem:

- the successful final directory retained the staging directory inode;
- the separate parent marker, not `COMPLETE.json`, was the commit authority;
- producer and independent exact-member/directory-root rehashes agreed;
- substitutions before the move, after the move but before the marker, and
  after marker linking all failed closed;
- `COMPLETE.json` without the parent marker failed closed;
- member/root mutation and marker-name substitution failed closed;
- the available `/proc/self/fd` path linked the held marker inode even after
  replacing the mutable fallback anchor.

### Requested v3-v5 regressions

The authenticated 57-test source suite passed (56 pass plus one expected
pre-manifest skip). Its covered regressions include:

- exact nine-part source/control symmetric closure: pipeline plus baseline
  plan, universal decoder, producer manifest, external audit bootstrap,
  extraction program, universal adapter, source snapshot, and source-free
  preflight;
- all eight complete matched-control pipelines and pre-fit closure rejection;
- exact unique ordered selector ordinals `0..149` in the preflight;
- final literal-container physical rate and independent decode/re-encode;
- explicit modeled-symbol density, unique pages, requested bytes, unique bytes,
  and repeated-requested-byte diagnostics;
- fail-closed fold/resource gates;
- UUID/PCI emitter canonicalization and cross-record identity equality;
- the decision triplet binding bits, levels, and base-frequency bytes;
- E250 unequal shapes, high owner indices, and shared tails, plus 128/256
  boundary portability.

The authenticated source-free RTX 5090 receipt independently contained 150
unique cell records with exact ordinals `0..149`, repeated CPU/CuPy equality,
canonical identity, and a committed parent-marker/directory-root pair. These
are implementation/reproducibility facts, not model evidence.

## Blocking findings

### B1. Verification returns no safely consumable authenticated result bytes

[`result_envelope.py`](../unifilar_wfa_entropy_census_stage0_v6/result_envelope.py)
opens and hashes all declared members, but captures only `COMPLETE.json`
(lines 182-193). It returns metadata only (lines 226-236) and closes every held
member, directory, and marker descriptor in `finally` (lines 237-243).

That contradicts the producer's own
[`INDEPENDENT_BOOTSTRAP_ABI.md`](../unifilar_wfa_entropy_census_stage0_v6/INDEPENDENT_BOOTSTRAP_ABI.md)
requirement at lines 117-122 to consume only held or already-buffered
authenticated bytes. After a successful call, a caller that opens
`GPU_DEV_RECEIPT.json` or a future result by pathname is resolving a new name,
not consuming the object that was hashed. Replacing the final directory after
verification and before that reopen is a verification-to-use race.

Required repair and gate:

1. Expose a context-managed verified result retaining parent, marker, directory
   and member descriptors, or run a consumer callback while those descriptors
   remain held. Buffering bounded small records is also valid.
2. Add a hostile test that substitutes the final directory after verification
   but before result parsing. The supported consumer API must still read the
   authenticated inode or fail; pathname reopen is forbidden.

### B2. Marker bytes can change after linking without producer detection

[`uwfa_common.py`](../unifilar_wfa_entropy_census_stage0_v6/uwfa_common.py)
opens the marker descriptor with `O_RDWR` (lines 903/917), verifies exact bytes
before linking (lines 1225-1242), then links it. Its post-link checks at lines
1263-1275 call `_verify_named_commit_binding`, which compares only regular-file
type and `(st_dev, st_ino)` (lines 1071-1080). `fchmod(0400)` does not revoke
write access already held by the open `O_RDWR` descriptor.

A hostile link callback can overwrite bytes through that descriptor after the
link while preserving inode and size. `complete()` can then return success even
though an independent consumer rejects the marker seal. The present tests cover
name/inode substitution, not content mutation through the held descriptor.

Required repair and gate:

1. After link, anchor cleanup and parent `fsync`, re-read the complete held
   marker, compare its SHA-256/length/content with `marker_bytes`, and recheck
   the named inode. Require the intended final link count after fallback-anchor
   cleanup.
2. Add a hostile test that writes through `source_fd` inside the link callback
   after creating the named marker. Producer completion must fail.

### B3. Inner validation and held-out selection are not charged by literal physical bytes

The final emitted container has a real physical ledger and the corresponding
tests pass. The scientific selector does not use that ledger.

[`stage0_census.py`](../unifilar_wfa_entropy_census_stage0_v6/stage0_census.py)
`validation_score` at lines 610-611 charges each logical stream only to its next
byte and adds the raw serialized model length. Nested candidate selection uses
that function at lines 1315-1319, and held-out savings use analogous byte-rounded
payload plus raw model accounting at lines 1322-1334.

The literal codec instead rounds every frame to 64 bytes (container lines
302-304) and places model, directory and regions on 4 KiB boundaries (lines
507-511). Therefore two candidates can be ordered differently by the selector
and by actual transmitted bytes, especially around 64-byte stream and 4 KiB
model/region boundaries. Calling the current value `validation_charged_bits`
does not make it physical.

Required repair and gate:

1. Select and score from an exact closed-form reconstruction of the literal
   packet layout, or build/decode the literal validation/test packet for every
   candidate. Include frame headers, per-frame 64-byte padding, shared model,
   directory, regions, 4 KiB alignment and ownership allocation.
2. Add adversarial candidate pairs on both sides of every 64-byte and 4 KiB
   boundary and assert that selector order equals literal container order.
3. Keep final physical bpw as `8 * literal_container_bytes / source_weights`.
   A source-free fixture's bits/symbol is never bpw unless the authenticated
   production ledger proves exactly one modeled symbol per source weight.

### B4. GPU telemetry validation is not bound to the workload it authenticates

The recorded source-free receipt is internally numerically consistent: its
182,736,318 count updates plus 41,629,938 length updates equal the claimed
224,366,256 measured updates. The source validator does not require that
equality.

[`stage0_census.py`](../unifilar_wfa_entropy_census_stage0_v6/stage0_census.py)
derives exact representative measured/projected update counts at lines 487-512.
It then validates phase telemetry only by H2D category conservation, kernel
subcount conservation, nonzero model H2D/D2H and positive peaks (lines
514-536). It never binds count/length updates, kernel counts, pack/subset calls
or host transfers to the derived workload. An honestly resealed receipt can
claim the full update count while reporting zero phase kernels/updates and only
minimal nonzero model-H2D/D2H, and pass. The all-150 environment telemetry has
the same missing semantic join.

The production validator also applies only the generic identifier grammar to
UUID and PCI values (lines 148-169). The emitter canonicalizes them, but an
honestly resealed CUDA and independent pair both containing `"x"` passes the
validator's equality test.

Required repair and gate:

1. Derive and enforce exact update, count-kernel, length-kernel, pack, subset,
   D2H, H2D and launch totals from authenticated cell/stream geometry for both
   representative and all-150 receipts.
2. Add an honestly resealed zero-work/minimal-transfer negative receipt; reject
   it before source access.
3. Enforce canonical GPU UUID and PCI regexes in the receipt validator itself,
   and add a resealed equal-but-noncanonical negative test.

### B5. The confidence interval treats dependent owner folds as iid samples

In exact-identity fold construction, a shared stream is in every owner's test
fold (`stage0_census.py` lines 1051-1058). Development folds also overlap
heavily. Lines 1359-1364 nevertheless feed per-owner saving values to an
ordinary Student-t interval with `df = fold_count - 1`.

Those values are not independent observations. Shared encoded bytes are
fractionally attributed across repeated views of the same stream, and the same
development data affect multiple selected models. The resulting t margin is
not a valid held-out confidence claim. Current fold regressions do not include
a dominant all-owner shared stream that exposes this pseudo-replication.

Required repair and gate:

1. Define genuinely independent clusters (normally held-out whole layers or
   disjoint model/family units) and compute uncertainty only across those
   clusters. If fewer than two independent clusters exist, mark confidence
   non-estimable and block promotion.
2. Add a shared-stream-dominant test; duplicating owners of one underlying
   stream must not shrink the confidence interval or increase degrees of
   freedom.

## Exact disposition of requested properties

| Property | Disposition |
|---|---|
| Local/RunPod byte authentication before execution | PASS |
| Full authenticated source suite | PASS, but incomplete against B1-B5 |
| Final directory equals retained staging inode | PASS in independent hostile transaction |
| Separate parent marker is sole commit authority | PASS for tested name/inode cases |
| Exact member and directory-root rehash | PASS |
| Three requested directory-entry fault windows | PASS |
| `/proc/self/fd` marker link identity vs mutable anchor | PASS on this RunPod |
| Consumer rejects `COMPLETE` without marker | PASS |
| Consumer rejects inode/root/member mismatch | PASS |
| Safe authenticated handoff into a result consumer | **BLOCK B1** |
| Marker content immutable through completion return | **BLOCK B2** |
| Source/control nine-part closure | PASS in source regressions |
| Exact unique preflight ordinals `0..149` | PASS in source suite and authenticated receipt |
| Final emitted physical bpw ledger | PASS in source regressions; no payload result run |
| Physical nested selector/held-out ledger | **BLOCK B3** |
| Explicit symbol density/read diagnostics | PASS in source regressions; no production values claimed |
| Fold construction/uncertainty | Construction passes; **confidence BLOCK B5** |
| Static resource admission | PASS for tested paths |
| Receipt telemetry-to-work binding | **BLOCK B4** |
| UUID/PCI receipt equality | PASS; canonical validator remains blocked under B4 |
| Decision triplet binding | PASS |
| E250 unequal/shared-tail regression | PASS |

## Smallest dependency-ordered repair/retest sequence

1. **Close artifact authority first:** repair B1 and B2; add verify-to-consume
   substitution and post-link marker-write hostile tests. Re-run publication
   tests and the independent harness.
2. **Close preflight authority:** repair B4; run the negative resealed telemetry
   and noncanonical identity tests, then repeat the source-free RTX replay and
   independently rehash its parent marker/directory.
3. **Close the scientific ledger:** repair B3 and its 64-byte/4 KiB boundary
   tests. Require all inner selections and test savings to match literal packet
   bytes.
4. **Close statistical promotion:** repair B5 using independent clusters and
   the dominant-shared-stream regression.
5. **Re-authenticate a new immutable source inventory and rerun the complete
   source suite.** Only after all four preceding stages pass is a manifest/freeze
   or any Qwen payload execution scientifically admissible.

No optional codec expansion is needed to decide this audit: v6 remains blocked
at source level until these gates pass.
