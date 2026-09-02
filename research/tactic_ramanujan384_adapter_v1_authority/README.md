# TACTIC Ramanujan-384 authority adapter v1

Date: 2026-09-02

Status: **frozen source-only; no Qwen, coarse-model payload, or production
execution authority**.

This sibling repairs every authority gap recorded by the independent audit of
`tactic_ramanujan384_adapter_v0` without changing v0.  It does not claim that
Ramanujan refinement works on Qwen.  Its purpose is to make a later payload
result literal and falsifiable.

## What v1 changes

1. `manifest.py` is the only source-root implementation.  Both freezing and
   verification sort member rows by name and serialize every object with
   sorted keys, compact separators, and finite JSON.  The producer/verifier
   root mismatch found in v0 is therefore structurally removed.
2. `adapter.py` emits a composite to a new regular file, reads it through an
   instrumented file reader, decodes the coarse payload using an explicit
   coarse-decoder capability, decodes all fine packets from the replayed
   container, and reconstructs literal weights.  It independently recomputes
   source-domain FP64 MSE and `F` from those weights.  The authenticated
   independent FP32 coarse-reconstruction hashes must equal the output of the
   literal coarse decoder for every role.
3. `codec_authority.py` packet-encodes and packet-decodes every representable
   rank candidate before that candidate can win.  The selected concatenated
   stream is decoded once more.  Projection energy and unreplayed floating
   corrections cannot select a winner.
4. `stable_controls.py` specifies SplitMix64 and a fixed twelve-`uint16`
   Irwin--Hall normal approximation.  Control FP64 arrays are made on the host
   and copied byte-for-byte to NumPy or CuPy; no backend RNG is called.
5. `authenticated_io.py` requires actual input-manifest and auditor
   source-manifest paths.  Both files are opened, duplicate-key parsed, and
   hashed against both the input binding and independent receipt.
6. `read_trace.py` separates two statements:
   * the contiguous minimal object has a layout upper bound of 1x;
   * the authority replay performs one instrumented data read totalling 1x.

   The trace is not physical storage telemetry and is not an HBM measurement.
7. `contract.py` explicitly defines the universal geometry domain.  Gate, Up,
   and Down-transposed use `[intermediate, hidden]`; dimensions are positive
   uint32 values; checkpoint/layer/expert identity is absent; exact
   `307/128`-bpw coarse length must be integral in bytes.  Partial 4096-value
   tails are charged in the fine stream, but nonexistent padded coordinates
   are never included in MSE or controls.  Actual page-rounded rate determines
   target eligibility.

The literal coarse decoder is intentionally an explicit dependency rather
than an encoder-side reconstruction shortcut.  A production pilot must bind
an independently audited coarse decoder; this source-only package provides
only a zero-coarse synthetic fixture.

## Pinned parents

The package pins the full v0 source manifest and root, the known v0
sorted-key root mismatch, every reused v0 Python member, and the independent
audit manifest/root/disposition in `dependency_lock.json`.  Neither parent
directory is modified.

## Source-only commands

```bash
python -I -B research/tactic_ramanujan384_adapter_v1_authority/test_source_only.py
python -I -B research/tactic_ramanujan384_adapter_v1_authority/run_source_free_fixture.py
python -I -B research/tactic_ramanujan384_adapter_v1_authority/verify_source.py \
  --package research/tactic_ramanujan384_adapter_v1_authority \
  --manifest-sha256 <expected-manifest-sha256>
```

The CuPy command has no payload argument or path-discovery surface:

```bash
python -I -B research/tactic_ramanujan384_adapter_v1_authority/run_source_free_cupy_smoke.py \
  --authorization RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_AUTHORITY_CUPY_V1 \
  --manifest-sha256 <expected-manifest-sha256>
```

A subsequent independent auditor must still review v1 before any Qwen launch.
This package contains no payload CLI and grants no payload authority.
