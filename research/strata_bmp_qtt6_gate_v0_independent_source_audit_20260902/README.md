# Independent source audit: STRATA-BMP/OBDD/QTT6 v0

Date: 2026-09-02

Externally pinned producer source:

- `SOURCE_MANIFEST.json` SHA-256:
  `a7778080a00d5d2967636ac8d60dd31698401c4dcf8da160c9451c92dc5f6b18`
- source-root SHA-256:
  `6b7baf9706349d10108121d4dcb03661b2378dc436303bbfe1bbccd38a0c8914`

The audit is payload-free. It must not open, enumerate, stat or hash Qwen,
STRATA, coarse-code or matched-control payloads. It authenticates the exact
ten-member frozen producer source before importing any producer module.

## Static conclusion

The implementation has the intended six completed planes and literal
`D[N,64]` interface. The accepted Qwen-compatible fixture maps 4,096 tile
coordinates bijectively to twelve decoder-derived bits, and all three packet
families charge literal byte lengths including selectors, padding, exceptions
and CRC. Label updates use exact conditional 64-way costs and every completed
candidate is re-decoded and re-scored with `D + lambda_bit * physical_bits`.
The synthetic and one empirical-moment-matched Gaussian fixture invoke the
same complete bounded search callable.

Two producer corrections are required before a production adapter is
authorized:

1. **Semantic canonicality is not established.** Byte-identical re-encoding
   only canonicalizes the parsed syntax. A zero rank-0 GF(2) factor packet and
   a zero rank-1 factor packet decode to the same index field. Swapping two
   factor columns gives distinct equal-rate packets for the same function.
   Zero rank-1 and rank-2 QTT packets also decode identically. ROBDD reduction
   is canonical for its fixed order, but the GF(2) factor and QTT gauges are
   not. This does not make bytes free--every alias is still charged--but it
   contradicts the stronger no-alias protected-property wording and would
   allow unused gauge bits to act as a hidden channel.
2. **The geometry validator is broader than the packet ABI.** It accepts
   `cols=65536` and `rows=98304=3*2^15`, while the header stores both as
   `uint16`. Encoding then raises `struct.error`, not the codec's fail-closed
   `CodecError`. The accepted universal shape family must be bounded to the
   serialized range, or the header widened.

Additional expected holds remain:

- the encoder search is NumPy; the CuPy path covers generated distortion,
  plane assembly and GF(2) matrix-product smoke primitives only;
- the producer smoke imports `cupy` without authenticating module/build/device
  identity and records device zero rather than the active device;
- the 64 MiB CPU cap is an input-size guard, not a measured peak-allocation
  receipt, and the search-operation counter is an abstract decision counter;
- the OBDD mechanism maximum is 2.83203125 bpw before production fields, and
  no complete-codec rate cap is enforced here;
- scale, RHT/KLT, profile, expert/component framing, page padding, independent
  original-BF16 scoring, at least eight fully selected Gaussian controls and
  routed cold-page reads are absent by design.

Therefore the static disposition is:

```text
HOLD_PRODUCTION__CORRECT_SEMANTIC_CANONICALITY_AND_UINT16_GEOMETRY__
THEN_SEPARATELY_BIND_STRATA_CONTROL_SCORER_AND_READ_LEDGER
```

This is not a negative result for label-flexible GF(2), ROBDD or QTT coding.
It is not Qwen evidence and makes no `F<=0.8`, 2.15--2.5 bpw or `<2x` read
claim.

## Execution state

The provided RunPod endpoint refused the audit SSH connection on 2026-09-02.
Consequently the hostile suite, N=4096 fixture and real-CuPy audit remain
unexecuted in this audit directory. Their source is frozen for replay, but
their presence is not a PASS receipt.

When the pinned endpoint is available, run in an isolated interpreter:

```bash
python -I -B run_audit.py \
  --source ../strata_bmp_qtt6_gate_v0 \
  --output hostile_audit_receipt.json
python -I -B ../strata_bmp_qtt6_gate_v0/run_source_free_fixture.py \
  > source_free_fixture_stdout.json
python -I -B run_real_cupy_audit.py \
  --source ../strata_bmp_qtt6_gate_v0 \
  --output real_cupy_audit_receipt.json
```

No execution receipt should be added unless the command exits zero and its
literal output is hashed into a new immutable receipt manifest.
