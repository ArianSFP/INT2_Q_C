# Independent source audit: STRATA-BMP/OBDD/QTT6 replay v2

Date: 2026-09-02

Externally pinned producer source:

- `SOURCE_MANIFEST.json` SHA-256:
  `84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8`
- source root:
  `b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47`

This review is source-only. It does not connect externally or open, enumerate,
stat or hash any Qwen, STRATA, coarse-code or matched-control payload. It does
not edit the producer and grants no payload or production authority.

## Outcome

The v1 replay and workspace findings are repaired.

- All thirteen non-manifest members are pinned, and their manifest order is the
  same ascending UTF-8 byte order enforced by `verify_source.py`.
- The canonical member JSON hashes exactly to the externally pinned source
  root. The documented CLI uses the parser's required `--package` and
  `--expected-manifest-sha256` arguments. A source test invokes that shape in a
  fresh isolated interpreter.
- `logical_capacity_plan` is explicitly a conservative capacity calculation,
  not runtime ownership or an allocator peak.
- Candidate maxima are derived from literal geometry and family semantics. The
  sixteen-member family bank correctly handles skew tiles whose BMP packets can
  exceed 2,048 bytes.
- Each instrumented stable ordering uses the actual `numpy.intp` dtype and
  `nbytes`. Retained packet objects are charged by literal packet length with
  acquisition/release events.
- Host logical capacity, instrumented host objects and the measured dedicated
  CuPy pool are separate receipt fields. No cross-allocator process peak is
  claimed.

The unchanged packet mechanism retains the exact six-plane `D[N,64]` ABI,
canonical minimum-rank GF(2) matrix/QTT gauges, reduced fixed-order ROBDD,
uint16 pre-pack geometry checks, literal packet formula and exact integer
2.15--2.5 bpw boundary.

## Findings and holds

### B1 — one harmless inventory-count typo

The README calls the source inventory “twelve-member”; the manifest contains
thirteen members excluding `SOURCE_MANIFEST.json`. The actual manifest,
verifier required set, source-root calculation and filesystem closure all agree
on thirteen, so self-replay is not affected. Correct the prose in a future
sibling, but this does not hold source-only execution.

### B2 — byte authentication is not a trusted production capability

`ArtifactBinding` materially improves v1: it rejects bad digests, symlinks and
non-regular objects and checks stable pre/post metadata. The control, read and
audit JSON parsers also enforce their small schemas.

However, `ProductionHooks.authorize()` still returns `authorized: true` for
caller-created dummy binary objects and caller-authored receipts. The
producer's own positive test demonstrates exactly this. In particular:

- `expected_source_manifest_sha256` and `expected_source_root_sha256` are
  arbitrary caller inputs, not the frozen producer pins;
- the “independent audit” receipt merely echoes those caller inputs and
  `passed: true`; it binds no trusted audit-manifest digest, executed test count,
  CuPy receipt or reviewer trust anchor;
- control receipts self-assert two booleans and an ID, but do not bind the
  control factory, codec packet, source/model manifest, selected transform,
  scorer, seed or resulting reconstruction;
- the routed-read receipt self-asserts one amplification number without binding
  framing, packet bytes, page accounting or expert identity;
- fixed binary objects are byte-authenticated relative to digests supplied in
  the same call, but their semantic formats and mutual compatibility are not
  checked.

This is integrity relative to caller-provided hashes, not independent
authorization. It is compatible with v2's explicit
`production_launch_authorized: false` and separate-launch-review hold, but a
downstream launcher must not treat this method's `authorized` field as a trust
decision by itself. Production needs an externally pinned launch manifest or
signature that binds the actual v2 source pins, audit source/receipt, controls,
scorer, packet and read ledger.

### B3 — runtime remains pending

The audit freezes an independent CPU suite and fresh-CuPy validator. Their
presence is not an execution receipt. Numerical fixture, self-replay, real GPU
search and independent reference closure remain pending until run in the
intended environment and immutably pinned.

## Disposition

```text
PASS_V1_REPLAY_AND_WORKSPACE_REPAIRS__
PASS_SOURCE_ONLY_MECHANISM_FOR_RUNTIME_REPLAY__
HOLD_TRUSTED_PRODUCTION_CAPABILITY__HOLD_RUNTIME_RECEIPTS__HOLD_PAYLOAD
```

This is not Qwen evidence, an `F<=0.8` result, a complete 2.15--2.5 bpw codec,
or evidence of routed reads below 2x.

## Replay

Independent CPU suite:

```bash
python -I -B run_audit.py \
  --source ../strata_bmp_qtt6_gate_v2_replay \
  --output cpu_audit_receipt.json
```

Independent real-CuPy validation:

```bash
python -I -B run_real_cupy_audit.py \
  --source ../strata_bmp_qtt6_gate_v2_replay \
  --output real_cupy_audit_receipt.json
```

Audit inventory:

```bash
python -I -B verify_audit_source.py \
  --package . \
  --expected-manifest-sha256 AUDIT_MANIFEST_SHA256
```

