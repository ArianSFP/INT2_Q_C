# TACTIC Ramanujan-384 adapter v0

Date: 2026-09-02

Status: **frozen source-only adapter; no Qwen or coarse payload execution
authority**.

This package is the finite, cheapest-first follow-up to the non-dyadic
Ramanujan oracle in `research/mosaic_secondary_oracles_v0`.  It does not
claim a Qwen result.  It binds the audited oracle source by SHA-256 and turns
the previously continuous screen into a literal 48-byte refinement record for
every 4,096 source weights.

## What is new

The public dictionary covers every non-power-of-two period from 3 through
127.  It uses exact integer shifts of the Ramanujan sum instead of a
platform-dependent trigonometric QR at decode time.  The construction spans
the same exact-period spaces as the primitive-frequency construction audited
in the parent package, while keeping the finite decoder procedural and small.

For each 4,096-value residual block the encoder performs a deterministic
top-correlation screen, batched ridge-stabilised least squares for ranks
1--14, FP16 scale rounding, signed 11-bit coefficient rounding, and exact
FP64 analysis-by-synthesis scoring in the original source coordinates.  Rank
zero is included.  No projection-energy shortcut selects the winner.

The `RPF0` record is exactly 384 bits:

| field | bits |
|---|---:|
| magic | 16 |
| version | 4 |
| role | 2 |
| rank | 4 |
| shared binary16 coefficient scale | 16 |
| up to 14 `(9-bit atom, 11-bit signed coefficient)` entries | 280 |
| canonical zero padding | 30 or more |
| CRC32 over the first 44 bytes | 32 |
| **total** | **384** |

Support, scale, coefficients, header, CRC and padding all displace refinement
bits.  There is no free period selector, support oracle, amplitude or model
table.

## Exact TACTIC ledger

The coarse plus fine payload is already below the cap:

```text
coarse       307/128 bpw
fine          12/128 bpw  (48 bytes / 4096 values)
payload      319/128 bpw = 2.4921875 bpw
```

`container.py` emits and authenticates a literal 512-byte expert header; it
contains the universal shape, all three block counts, byte lengths, the full
coarse and fine SHA-256 values, the source-binding SHA-256, CRC32 and canonical
zero padding.  For the audited 768x2048x3 geometry the complete object is
1,470,464 bytes (359 pages), or `2.4930555555555554 bpw`.  Thus every outer
byte is charged and the object remains below 2.5 without consuming the older
4,608-byte / 1/128-bpw TACTIC header allowance.

The composite is expert-local, page aligned, fetched once and buffered, so
external-storage read amplification is exactly `1.0`.  HBM traffic is not
measured and this package makes no accelerator-HBM claim.  Shapes with a
partial 4,096-value role tail are supported mechanically, but their literal
padding is charged and they are not target-eligible if physical rate exceeds
2.5 bpw.

## Authentication boundary

`authenticated_io.py` refuses to open a production role unless all of the
following are supplied and agree:

* an expected binding-file SHA-256;
* the independent coarse-audit receipt and its pinned SHA-256;
* the literal coarse artifact and its independently recorded hash;
* the canonical BF16 source and its independently recorded role hash;
* an independently decoded FP32 coarse reconstruction and its recorded hash;
* the input manifest and independent auditor source-manifest pins.

The normalized universal role geometry is always
`[intermediate, hidden]` for `gate`, `up`, and `down_transposed`.  Checkpoint,
layer and expert identity are not codec inputs.

`adapter.py` is the source-frozen whole-expert bridge.  It requires all three
authenticated roles in canonical order, proves they share one coarse object,
pools the literal original-domain score, emits and immediately replays one
container, checks the physical rate before promotion, and (only after an
absolute source pass) reruns every role under all nine controls.  There is no
payload CLI in this unaudited source package.

## Controls and promotion rule

The absolute source must first reach `D <= 0.025`.  Only then is the complete
rank/scale/packet search rerun on:

1. one frozen odd-affine within-block phase destruction; and
2. eight blockwise mean-and-centred-energy matched Gaussian controls.

The source gain must exceed the strongest of all nine controls by at least
0.03 bpw.  Control subtraction cannot rescue an absolute miss.  A one-expert
survivor is only eligible for an independently audited payload pilot; a
universal claim additionally requires whole-layer/whole-expert uncertainty
and sealed transfer to a disjoint SwiGLU-MoE family.

## Source-only checks

```bash
python -I -B research/tactic_ramanujan384_adapter_v0/test_source_only.py
python -I -B research/tactic_ramanujan384_adapter_v0/run_source_free_fixture.py
python -I -B research/tactic_ramanujan384_adapter_v0/verify_source.py \
  --package research/tactic_ramanujan384_adapter_v0 \
  --manifest-sha256 <expected SHA-256>
```

The CUDA smoke has no payload arguments:

```bash
python -I -B research/tactic_ramanujan384_adapter_v0/run_source_free_cupy_smoke.py \
  --authorization RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_CUPY_SMOKE_V0 \
  --manifest-sha256 <expected SHA-256>
```

RunPod execution is pending while the provided endpoint is unavailable.  No
payload launch is authorized by this source freeze.
