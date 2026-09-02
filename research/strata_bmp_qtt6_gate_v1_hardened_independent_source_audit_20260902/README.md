# Independent source audit: STRATA-BMP/OBDD/QTT6 hardened v1

Date: 2026-09-02

Externally pinned producer source:

- `SOURCE_MANIFEST.json` SHA-256:
  `916aaca15620e3bf033e849b74a73604015fab280dfe8953683d6cbe04e0d2e4`
- producer source root:
  `369e01b30173977a5d8227e71104c8515f1b68ef440198dccd1488050e865203`

This is a payload-free, benign correctness and reproducibility audit. It does
not open, enumerate, stat or hash Qwen, STRATA, coarse-code or matched-control
payloads, and it grants no production-launch authority. The audit authenticates
the exact twelve-member producer package before importing any producer module.

## Static conclusion

The two blocking v0 defects are repaired:

1. The packet geometry is checked against the six literal uint16 fields before
   serialization. Shapes at 65,535 round-trip, while any serialized field at
   65,536 fails with `CodecError` instead of leaking a `struct.error`.
2. A GF(2) matrix plane is encoded only at its exact minimum rank in a fixed
   earliest-column/earliest-row gauge. A QTT plane is decomposed by exact,
   deterministic GF(2) rank factorizations at every cut; zero has one empty
   representation. Rank inflation, column/bond gauges and unused rank-mask bits
   are rejected by semantic re-derivation.

The six completed planes, literal `D[N,64]` ABI, mixed-radix public geometry,
canonical ROBDD construction, exact packet-byte rate formula, upper/lower
integer rate bounds and fail-closed production hooks are internally consistent.
The CuPy source performs a real rank-0/rank-1 BMP alternating search on device;
it is not merely a device-generated distortion-table smoke. The isolated worker
records module/distribution/device identity and dedicated-pool samples. ROBDD
and QTT GPU search correctly remain explicit holds.

## New findings

### A1 — the declared logical-workspace plan is not an exact runtime ledger

`exact_workspace_plan` declares `32 * 2,048` bytes for the candidate packet
bank. A valid production geometry may be a skew tile such as `4 x 1,024`.
A canonical rank-four BMP component then occupies
`ceil(4*(4+1024)/8) = 514` bytes. Six planes, ranks, header/CRC and 64 exceptions
can occupy 3,316 bytes for one candidate. Likewise a `1 x 4,096` rank-one tile
already exceeds 2,048 bytes. The complete current family bank has only four BMP
candidates, four OBDD candidates and eight QTT candidates, so its maximum
aggregate packet payload still fits inside the conservative 64 KiB declaration.
There is no packet-bank allocation in the implementation, however, and 2,048
is not a geometry-independent per-candidate maximum. Thus the field is a design
capacity rather than an exact account of owned runtime packet bytes. The
independent suite freezes a constructive 3,316-byte packet witnessing that
distinction; it does not claim the overall 64 MiB cap is exceeded.

The independent test also records that NumPy's stable `argsort` index array is
normally platform `intp` (eight bytes on the RunPod), whereas the named plan
calls the buffer `stable_order_i32` and charges four bytes per entry. The
producer explicitly disclaims an allocator peak, but this second mismatch
reinforces that the plan is a design estimate rather than an exact inventory of
the search's live logical buffers.

Required correction before any exact-workspace claim: publish this as a
conservative design-capacity ledger, or derive actual live byte ownership from
the implementation; and either force a real int32 ordering buffer or charge
`np.intp`.

### A2 — the frozen source verifier cannot replay its own manifest

The README invokes `verify_source.py --manifest ...`. The frozen verifier
defines only `--package` and `--expected-manifest-sha256`; it has no
`--manifest` argument.

There is a second independent blocker. `verify_source.py` requires manifest
members to equal `sorted(names, key=lambda value: value.encode("utf-8"))`.
The frozen member list begins with lowercase `codec.py` and places uppercase
`README.md` and `THREAT_MODEL.md` later, so it is not in the required bytewise
order. The source root was computed over that listed order. Consequently even
the syntactically correct command below fails the verifier's `canonical complete
members` condition for the externally pinned object:

```bash
python -I -B research/strata_bmp_qtt6_gate_v1_hardened/verify_source.py \
  --package research/strata_bmp_qtt6_gate_v1_hardened \
  --expected-manifest-sha256 \
  916aaca15620e3bf033e849b74a73604015fab280dfe8953683d6cbe04e0d2e4
```

Both the option and manifest/verifier ordering contract require correction in
a new frozen sibling. This is source-authentication and reproducibility debt,
not a codec-semantic failure; the independent audit authenticates the exact
external manifest and each member without imposing the producer's inconsistent
ordering predicate.

### A3 — production hooks are syntactic commitments, not launch evidence

`ProductionHooks.authorize()` correctly fails when any required digest is
absent or fewer than eight controls are declared. Conversely, any nine strings
matching the lowercase 64-hex syntax and a count of eight pass. It does not
open and authenticate the referenced objects, prove that controls repeat the
complete selection procedure, bind a source/model manifest, or execute the
read ledger. This is consistent with the producer's separate-launch-review
hold. The independent suite preserves that distinction.

### A4 — runtime remains pending in this audit source freeze

This audit directory contains an independently pinned 21-test CPU suite and a
fresh-CuPy receipt validator. Neither is represented as executed by the source
manifest. A runtime PASS requires literal receipts from the pinned source on a
Python/NumPy environment and the CuPy/CUDA RunPod. Static source presence is not
a runtime result.

## Disposition

```text
PASS_V0_SEMANTIC_CANONICALITY_AND_UINT16_REPAIRS__
HOLD_EXACT_WORKSPACE_LEDGER_AND_REPLAY_DOC_CORRECTIONS__
HOLD_RUNTIME_CUPY_RECEIPT__HOLD_ALL_PAYLOAD_AND_PRODUCTION_AUTHORITY
```

This is not a Qwen result, an `F<=0.8` result, a complete 2.15--2.5 bpw codec,
or evidence of routed reads below 2x. The mathematical BMP/ROBDD/QTT mechanism
may proceed to runtime source-only replay after the ledger finding is either
fixed or explicitly downgraded; production payload launch remains separately
held by the producer's own contracts.

## Replay

CPU source-only suite:

```bash
python -I -B run_audit.py \
  --source ../strata_bmp_qtt6_gate_v1_hardened \
  --output cpu_audit_receipt.json
```

Fresh CuPy worker plus independent receipt validation:

```bash
python -I -B run_real_cupy_audit.py \
  --source ../strata_bmp_qtt6_gate_v1_hardened \
  --output real_cupy_audit_receipt.json
```

Audit-source inventory:

```bash
python -I -B verify_audit_source.py \
  --package . \
  --expected-manifest-sha256 AUDIT_MANIFEST_SHA256
```
