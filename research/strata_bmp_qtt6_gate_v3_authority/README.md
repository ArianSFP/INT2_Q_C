# STRATA-BMP/OBDD/QTT6 v3 evidence authority

Date: 2026-09-03

Status:

```text
FROZEN_SOURCE_ONLY_AUTHORITY__COMPILED_LAUNCH_PIN_ABSENT__HOLD_RUNTIME_PAYLOAD_AND_RESULTS
```

This narrow sibling repairs the independent review of
`strata_bmp_qtt6_gate_v2_replay` without modifying v2. It changes no BMP,
ROBDD, QTT6, exception, or candidate-search mathematics. Its only job is to
replace caller-authored JSON assertions with exact, independently pinned,
executed capability closures.

No Qwen, other model, current-STRATA packet, matched-control payload, or
RunPod endpoint was opened while building or testing this source package.

## Frozen lineage

- v2 `SOURCE_MANIFEST.json` SHA-256:
  `84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8`
- v2 source root:
  `b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47`
- independent v2 audit manifest SHA-256:
  `324e9a6d7d16be7b57b4ae33599cce2e4b324848e279b59268826b5dcaaebd12`
- independent v2 audit source root:
  `c817b1f1c3c270cb1f0e332262dc46df4fe9eb39c4b4fafe70a23536203572d3`

The v2 README's “twelve-member” wording was the audit's harmless B1 typo.
The actual v2 manifest and closure contain **thirteen non-manifest members**.
`authenticate_pinned_predecessors()` now opens the literal producer and
auditor manifests, hashes all thirteen and seven members respectively,
recomputes both roots, checks exact regular-file closures, and verifies the
auditor's producer pins. It never trusts a caller's spelling of those roots.

This v3 package itself has **eight non-manifest members**, all named in its
canonical source manifest.

## The authority discontinuity

V2 authenticated files against hashes supplied in the same call and accepted
self-authored control, audit, and read receipts. V3 has no such production
arguments. `authorize_production()` reads one compiled launch-manifest digest:

```text
TRUSTED_LAUNCH_MANIFEST_SHA256 = None
```

The source freeze intentionally leaves it `None`, so production authorization
fails before resolving an evidence or payload path. A later deployment sibling
may replace it only after an independent controller has frozen the complete
launch closure. Passing a newly invented digest to the lower-level verifier is
explicitly not production authority.

The one launch manifest pins six distinct executed capability directories:

1. predecessor source/audit replay;
2. Gaussian-control generation and complete-selection replay;
3. literal current-STRATA decoding;
4. independent original-BF16/FP64 scoring;
5. instrumented routed-expert page reads;
6. an independent replay of the complete launch evidence.

Each capability has its own canonical manifest, implementation bytes,
execution receipt, and independent audit receipt. The launch manifest pins its
manifest, member root, execution-receipt hash, and audit-receipt hash. V3 opens
and hashes every member, rejects links and extra entries, and requires distinct
producer, executor, and auditor identities. Production evidence marked
`dummy`, `test_fixture`, or `self_authored` is rejected. The package's hostile
test fixtures are visibly marked `SOURCE_TEST_FIXTURE`, `dummy=true`, and
`self_authored=true`; they cannot cross that
branch.

## Literal current-STRATA contract

The adapter capability cannot merely name STRATA. For every routed case it
must bind the actual packet hash and byte count and attest:

- the current six completed level-major planes and decoded index range 0..63;
- nonempty scale bytes inside the same literal packet;
- pinned forward-transform identity and forward/inverse implementation hashes;
- exact header, payload, trailer, and padding byte counts whose sum is the
  packet length;
- canonical packet decode/re-encode byte equality;
- decoded weight count and decoded reconstruction hash.

Those packet files are independently opened and hashed from the precommitted
evidence root. A four-level proxy, SC-decision stream, scale-free packet, or
unframed component cannot satisfy the ABI.

## Independent original-BF16 score

The scorer is a separate capability with different producer and executor
identities from the adapter. Every score row binds the three literal BF16 role
hashes and the adapter's decoded-reconstruction hash. V3 independently
recomputes

```text
relative_mse = FP64_SSE / FP64_source_energy
rate_bpw     = physical_packet_bits / source_weight_count
F            = relative_mse * 2**(2*rate_bpw)
```

and recomputes the pooled model score from routed rows. A production evidence
set must remain within 2.15--2.5 bpw and satisfy pooled model `F <= 0.8`.
Caller-supplied aggregate metrics are not accepted as authority.

## A read trace is not a layout ratio

Every model and control route has exactly one `(layer, expert)` packet and an
instrumented read trace. Each event records sequence, page index, file offset,
literal bytes read, and page hash. V3 recomputes the unique pages, physical
bytes, and

```text
cold_read_amplification = physical_page_bytes_read / literal_packet_bytes
```

for each routed expert and requires it to be strictly below `2x`. A receipt
with `layout_only=true`, a whole-layer aggregate, or a fabricated amplification
number fails.

## Model/control non-aliasing

The precommitted manifest lists every Gate, Up, and transposed Down BF16 file.
V3 resolves and opens each one, checks the byte count and SHA-256, and rejects
reuse by canonical path, `(device,inode)` identity, or content hash. This is
stricter than merely requiring different route IDs: a model and its matched
Gaussian control cannot be the same bytes under another name or hard link.
Production requires at least eight complete selected-control routes per model.

## Source-only replay

```powershell
python -I -B research/strata_bmp_qtt6_gate_v3_authority/test_source_only.py

python -I -B research/strata_bmp_qtt6_gate_v3_authority/verify_source.py `
  --package research/strata_bmp_qtt6_gate_v3_authority `
  --expected-manifest-sha256 MANIFEST_SHA256 `
  --v2-package research/strata_bmp_qtt6_gate_v2_replay `
  --v2-audit-package research/strata_bmp_qtt6_gate_v2_replay_independent_source_audit_20260902
```

The fourteen tests use only temporary, explicitly dummy and self-authored
source-test fixtures.
They test actual predecessor closure, compiled hold, complete fixture closure,
fixture/production separation, extra-file and receipt tampering, self-authored
receipts, literal scale bytes, independent FP64 arithmetic, layout/read
separation, recomputed `<2x` amplification, launch precommit, model/control
aliasing, and canonical JSON.

## Claim boundary

This is an executed source-test result for the authority mechanics only. It is
not a runtime capability, independent v3 audit, current-STRATA execution,
Qwen result, Gaussian-control result, `F<=0.8` result, or `<2x` measured model
read result. Runtime remains held until the six real capabilities execute,
are independently audited and pinned, and a later source closure compiles the
single launch-manifest trust root.
