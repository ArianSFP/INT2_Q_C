# STRATA-BMP/OBDD/QTT6 replay-safe source gate v2

Date: 2026-09-02

Status:

```text
FROZEN_SOURCE_ONLY__RUNTIME_REPLAY_PENDING__NO_PAYLOAD_AUTHORITY
```

This narrow sibling repairs the independent v1 audit findings without
modifying v1. It retains the v1 six-plane coordinate-function mechanism and
canonical packet semantics; it changes replay authentication, workspace
language and the production-capability boundary. It opens no Qwen, STRATA,
coarse-code or matched-control payload.

Pinned predecessor roots

- v1 source root:
  `369e01b30173977a5d8227e71104c8515f1b68ef440198dccd1488050e865203`
- producer manifest authenticated by the v1 audit:
  `916aaca15620e3bf033e849b74a73604015fab280dfe8953683d6cbe04e0d2e4`
- v1 independent-audit source root:
  `db0dea7fe3f52e88c8ab59af75eb7ceef71610c07f214fc5b2783f77dd98b56c`
- v1 independent-audit manifest:
  `c0c23ea892ed8066c9af78c15491049400a7f93f7f8c7d39b61b67442d10b0ed`

## Semantic ABI retained

The input remains a literal finite nonnegative `float64 D[N,64]` table. A
decoded index is assembled from exactly six completed planes:

```text
index[i] = sum(plane[level,i] << level for level in 0..5)
```

It is not a four-level proxy and not an internal SC-decision stream. Public
coordinates remain `role(base 3) x row x column`, with arbitrary positive
uint16 SwiGLU dimensions and a power-of-two local tile of at most 4,096
weights. BMP and GF(2)-QTT use canonical minimum-rank gauges; ROBDD uses its
fixed-variable-order reduced canonical form. Headers, selectors, model bytes,
exceptions, tails and CRC remain physical.

## Canonical UTF-8 replay

`SOURCE_MANIFEST.json` lists the complete twelve-member source inventory in
canonical ascending UTF-8 byte order. `verify_source.py` enforces the same
ordering, exact regular-file closure, every member SHA-256 and the canonical
source root. The source-only suite includes a manifest self-replay test that
executes the exact documented verifier CLI in a fresh isolated interpreter.

This fixes both v1 replay defects: the documented options now match the parser,
and the frozen manifest order matches the verifier predicate.

## Capacity is not runtime ownership

v2 separates three different quantities that v1 blurred:

1. `logical_capacity_plan` reports conservative logical capacities. It is not
   an allocator peak.
2. `candidate_serialized_capacity` derives every family candidate maximum from
   the serialized `row_count`, `col_count`, active feature count and family
   caps. There is no fixed 2,048-byte candidate fiction. A `1 x 4,096` rank-one
   BMP candidate is correctly larger than 2,048 bytes.
3. `WorkspaceLedger` records literal objects owned by the runtime instrumentation.
   Each `np.argsort(..., kind="stable")` result is charged using its actual
   `numpy.intp` dtype and `nbytes`; each retained packet is charged by
   `len(packet)`. Events include acquisition and release. This is exact for the
   named owned objects and explicitly not a Python/NumPy allocator peak.

The GPU receipt likewise keeps logical serialized capacities separate from the
measured CuPy pool. A fresh dedicated `cupy.cuda.MemoryPool` reports synchronized
used/reserved samples and a measured peak under the 128 MiB cap. v2 does not add
logical CPU estimates to a device-pool measurement or call the sum a process
peak.

## Object-authenticated production boundary

`production_hooks.py` no longer grants a capability because nine strings merely
look like SHA-256 digests. Every binding contains a literal `pathlib.Path` and
an externally supplied digest. Authorization performs stable pre/post `lstat`,
rejects symlinks and non-regular files, reads the object, and authenticates its
actual bytes.

The boundary additionally opens and semantically checks:

- at least eight distinct complete-selection Gaussian-control receipts;
- a routed cold-read receipt with maximum amplification strictly below `2x`;
- an independent-audit receipt bound to the exact producer manifest and source
  root.

The source package ships only `held_source_only_hooks()`. No production object
or receipt is bundled, so launch remains fail-closed. Object authentication is
necessary but is still not independent audit evidence.

## Physical rate and capability holds

`CompleteRateCap` continues to enforce exact integer bounds:

```text
ceil(43*N/20) <= complete physical bits <= floor(5*N/2)
```

Outer fields, prior packets and reserved bits are caller-owned and charged.
The fixture cannot call finalization because it has no authenticated outer
STRATA packet. ROBDD/QTT GPU production search, the current scale decoder,
forward/inverse transforms, BF16 scorer, full controls, expert framing,
read ledger and independent source audit all remain held.

## Replay

CPU source-only suite, including manifest self-replay:

```bash
python -I -B research/strata_bmp_qtt6_gate_v2_replay/test_source_only.py
```

Synthetic mechanism fixture:

```bash
python -I -B research/strata_bmp_qtt6_gate_v2_replay/run_source_free_fixture.py
```

Fresh CuPy worker:

```bash
python -I -B research/strata_bmp_qtt6_gate_v2_replay/run_cupy_smoke.py \
  --output cupy_search_receipt.json
```

Canonical source replay (the argument shape is exactly the parser contract):

```bash
python -I -B research/strata_bmp_qtt6_gate_v2_replay/verify_source.py \
  --package research/strata_bmp_qtt6_gate_v2_replay \
  --expected-manifest-sha256 MANIFEST_SHA256
```

## Claim boundary

This package is a source-only mathematical mechanism and replay hardening. It
is not evidence of Qwen gain, `F<=0.8`, a complete 2.15--2.5 bpw codec, or
routed expert reads below `2x`. No payload run is authorized by this freeze.
