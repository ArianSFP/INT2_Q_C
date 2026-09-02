# Independent assessment

## Disposition

`PASS_LOCAL_SOURCE_MECHANISM__HOLD_PRODUCTION_GLOBAL_PAYLOAD_AND_READS`

The frozen package contains a coherent source-only local mechanism: its six
level-major planes reconstruct the current STRATA 0..63 index ABI; its RM row
selection uses the orientation implied by the pinned independent STRATA
decoder; its packet has explicit bank/order/coset fields, CRC, canonical
arithmetic replay, zero alignment and a hard 1,280-byte local cap; and its
small-N exact search is a legitimate mechanism oracle.

That is not production authority. The package has no full joint RM(5,12)^6
encoder, no physical implementation or current-K rate for the global
RM-ordered alternative, no outer expert container, no inverse RHT/KLT path, no
source-domain scorer, and no routed-read benchmark. No Qwen, coarse, or matched
control payload was opened by this audit. Every payload claim remains `HOLD`.

## Independent static findings

1. `SRM6-AUDIT-001` — medium: `packet_ledger(bank, 9889)` correctly reports
   that the physical result would exceed 2.5 bpw, but its string disposition
   incorrectly calls that hypothetical packet
   `MECHANISM_FIXTURE_BELOW_2_15_BPW`. Literal `_build_packet` and
   `decode_packet` still reject it, so this is a public-ledger/status defect,
   not a cap bypass. It must be fixed before automated promotion consumes the
   string field.

2. `SRM6-AUDIT-002` — low: `exact_joint_oracle` applies `int(order)` and thus
   silently accepts a fractional order such as 1.5 while reporting the
   unmodified value in its receipt. This affects the bounded research oracle,
   not the fixed integer packet banks. It should fail closed on non-integers
   before any dominant-oracle result is admitted.

3. `SRM6-AUDIT-003` — blocking by design: the global helper orders phases by
   popcount, but it does not build a current 2^20/2^21 packet or measure its
   canonical arithmetic length. An arbitrary current K is correctly called an
   “RM-ordered truncated polar set,” not exact RM.

4. `SRM6-AUDIT-004` — blocking by design: the 40-byte local header has no
   expert identity, tensor role/shape, subblock ordinal/count, KLT coefficients,
   outer source binding, expert directory, or read schedule. The local decoder
   stops at six-plane indices; `scale` and `rht_seed` are returned but no inverse
   transform consumes them. Outer overhead is not reserved inside a packet at
   the exact 2.5-bpw cap. Production physical rate and read amplification are
   therefore unestablished.

5. `SRM6-AUDIT-005` — high: the CuPy smoke receipt reports only device name,
   CuPy version and CUDA runtime version. It does not bind the source manifest,
   source root, frozen STRATA auditor, Python executable, wheel trees, CUDA
   driver, device ID/UUID, or its own receipt hash. Its hard-coded PASS status
   also does not require `packet_fits_2_5_bpw`. A successful execution can be
   cited only as source-free mechanism evidence.

## Execution boundary at freeze

The audit source was frozen while `ssh root@74.2.96.53 -p 12079` returned
“connection refused,” and the local Windows sandbox had no Python interpreter
or WSL installation. Consequently, the detailed hostile suite and CuPy smoke
were not represented as executed at source-freeze time. `audit_strata_rm6_v0.py`
is frozen unchanged so it can be run once the RunPod returns; an execution
result is a separate artifact and must not rewrite this source package.

Static/manual checks completed at freeze include the three external hashes,
all source member hashes and closure, packet/header arithmetic, exact cap
boundary derivation, source-to-frozen-auditor SC schedule comparison, outer
field inventory, and the five findings above. They are auditable in the frozen
test code but are not mislabeled as a completed Python/CuPy run.
