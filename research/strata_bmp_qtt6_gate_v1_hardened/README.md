# STRATA-BMP/OBDD/QTT6 hardened source gate v1

Date: 2026-09-02

Status:

```text
FROZEN_SOURCE_ONLY__EXECUTION_PENDING__NO_PAYLOAD_AUTHORITY
```

This sibling repairs the two blocking findings in the independent static audit
of v0. It does not modify v0 and it does not claim a Qwen result.

Pinned predecessor evidence

- v0 `SOURCE_MANIFEST.json` SHA-256:
  `a7778080a00d5d2967636ac8d60dd31698401c4dcf8da160c9451c92dc5f6b18`
- v0 source root:
  `6b7baf9706349d10108121d4dcb03661b2378dc436303bbfe1bbccd38a0c8914`
- independent v0 audit manifest SHA-256:
  `6038fea16ba29fad6c8b351bc0968fd00f94f007f1e113a4209775974cc33df1`
- independent v0 audit source root:
  `e905324af56f544b27423390e22c97de5c4b15696c621ab133c2da5533e9f4a9`

## Exact semantic ABI

The input is a literal nonnegative finite `float64 D[N,64]` table. A decoded
index is assembled from exactly six completed planes:

```text
index[i] = sum(plane[level,i] << level for level in 0..5)
```

The gate does not accept a four-level proxy, a polar SC decision, or a
partially completed STRATA state.

The geometry is the public mixed-radix domain
`role(base 3) x row(base rows) x column(base cols)`. Rows and columns may be
arbitrary positive SwiGLU dimensions. Only the local algebraic tile remains an
aligned power-of-two rectangle with at most 4,096 values. This admits widths
such as 704x2304, 5760x3584 and 11008x4096 instead of only `3*2^k` by `2^h`.
Every serialized geometry field is checked against the exact uint16 range
before packing; invalid geometry raises `CodecError`.

## Semantic canonicality

### GF(2) matrix factor

For each decoded binary plane, `canonical_gf2_factor` computes the exact GF(2)
rank, takes the earliest independent matrix columns as the basis and fixes the
coordinates using the earliest independent rows. Encoding accepts only this
minimum-rank representation. Consequently all of these are rejected:

- rank-1 or rank-2 encodings of the zero plane;
- swapped factor columns;
- arbitrary `GL(r,2)` factor gauges;
- a descriptor above the rank-four semantic cap.

### GF(2) QTT

Each truth table is folded by its decoder-derived Boolean coordinate bits and
factorized at every cut by deterministic exact GF(2) rank factorization. A
nonzero function has one minimum TT-rank vector and one fixed core gauge. The
zero function has a dedicated zero rank code and no core bits. Encoding and
decoding reconstruct the truth table and require the supplied representation
to equal this canonical decomposition. Rank inflation, unused states, bond
gauge aliases and unused rank-mask bits fail closed.

The ROBDD family retains its fixed-variable-order reduced canonical form.

## Physical rate contract

All mechanism bytes remain literal: header, family/order selectors, canonical
model, exceptions, byte tails and CRC. In addition, `CompleteRateCap` is a
mandatory search argument. It uses exact integer arithmetic for:

```text
ceil(43*N/20) <= total physical bits <= floor(5*N/2)
```

The caller must charge outer fields, prior packets and reserved future bits.
A candidate above the remaining upper budget cannot win. `assert_complete`
requires the reservation to be zero and also rejects a final object below
2.15 bpw. The source fixture intentionally
does not call `assert_complete`: it has no authenticated STRATA outer packet
and therefore is not a complete-codec rate claim.

## Workspace and real CuPy path

The CPU search publishes every named logical buffer and its exact byte count;
the former 64 MiB input-size heuristic is no longer described as a measured
allocator peak.

`cupy_backend.py` contains an actual device-backed bounded rank-0/rank-1 BMP
search. Device work includes nearest-label selection, conditional six-plane
costs, alternating GF(2) factor decisions, index updates and exact candidate
SSE. `run_cupy_smoke.py` launches `cupy_worker.py` using a distinct
`python -I -B` process and authenticates a random nonce and PID. The worker
records:

- CuPy module origin and file hash;
- owning installed distribution and matching version;
- a compiled-kernel identity probe;
- active CUDA device id, name and PCI bus id;
- runtime and driver versions;
- a dedicated CuPy memory pool's exact used and total-reserved bytes at each
  synchronized checkpoint, with a hard 128 MiB cap.

The GPU backend is deliberately scoped to the rank-0/rank-1 BMP bank. ROBDD
and canonical QTT GPU search remain explicit holds; a partial accelerator is
not presented as a complete production search.

## Production holds

`production_hooks.py` defines the exact launch boundary. It fails closed until
all of the following are bound by SHA-256 and independently reviewed:

- current STRATA packet and scale decoder;
- forward and inverse RHT/KLT or other transforms;
- component/expert framing and page padding;
- original-BF16 scorer;
- one identical complete-selection Gaussian-control factory and at least eight
  selected controls;
- routed cold-read ledger;
- independent audit receipt.

No scale, transform, scorer, controls, expert packet or read ledger is bundled
here. No model checkpoint, STRATA payload, coarse artifact or matched-control
artifact is opened, enumerated, stated or hashed by this package.

## Replay

When a real Python environment is available:

```bash
python -I -B research/strata_bmp_qtt6_gate_v1_hardened/test_source_only.py
python -I -B research/strata_bmp_qtt6_gate_v1_hardened/run_source_free_fixture.py
python -I -B research/strata_bmp_qtt6_gate_v1_hardened/run_cupy_smoke.py \
  --output cupy_search_receipt.json
python -I -B research/strata_bmp_qtt6_gate_v1_hardened/verify_source.py \
  --manifest research/strata_bmp_qtt6_gate_v1_hardened/SOURCE_MANIFEST.json
```

The provided RunPod endpoint was refusing SSH while this sibling was frozen.
Therefore source tests, fixture and real-CuPy worker are marked execution
pending. Source presence is not a runtime PASS.

## Claim boundary

This is a source-only mathematical mechanism and hardened packet ABI. It is
not evidence of Qwen gain, `F<=0.8`, a complete 2.15--2.5 bpw codec, or `<2x`
routed reads. Payload work remains held until independent source audit and a
separate production-launch review both pass.
