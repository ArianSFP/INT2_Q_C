# SILT-INT2 source-free finite mechanism v1

Status: **implementation review only — no manifest, freeze, result, payload, or
source-gain authority**.

This is a format-breaking repair of v0. It closes the independent audit’s
mechanism blockers without reading any model weights. The architecture remains
a synthetic finite-label mechanism; it is not evidence that a dependency exists
in Qwen or any other SwiGLU-MoE.

## What v1 changes

- The expert directory is a separately hashed, page-aligned region. Expert
  counts `1..256` are explicit and runtime-tested, including 249, 250, and 256.
- Every expert carries its own bounded lane/vector geometry. All scalar fields,
  products, prefix sums, slices, factorial inputs, and decoded allocations are
  capped before work.
- GF(2) admits exactly six canonical selector IDs `0..5`; alias syntax `6/7` is
  rejected. Z4 retains all eight IDs. Three bits is the minimum fixed-width
  syntax for six choices.
- The arithmetic stream includes exactly 30 physical zero guard bits. Decode is
  bit-limited, must exhaust `meaningful_bits` exactly, and ordinary frame decode
  must reproduce the identical payload and length on re-encode.
- Cold ownership is `G/E + F_e`, not `T/E`. Decisions use exact integer
  cross-multiplication, owner shares sum to the literal container bytes, and an
  instrumented page reader independently reproduces `G+F_e`.
- CuPy search is mandatory and fails closed. Telemetry records exact logical
  transfer bytes, CUDA-event timings, host RSS baseline/peak/delta,
  baseline-subtracted NVML process/device VRAM, CuPy pool use, GPU UUID, PCI bus,
  CUDA logical index, NVML physical index, and the mapping assertion.
- A stdlib-only bootstrap authenticates an externally supplied source root
  before any sibling or third-party import, copies those exact bytes to a
  private snapshot, and invokes Python in isolated mode.
- Runtime output uses a hidden staging directory, exclusive no-follow files,
  file/directory `fsync`, atomic `renameat2(RENAME_NOREPLACE)`, parent `fsync`,
  and descriptor-based post-publication rehashing.

## Explicit limits

The frozen source contract caps experts at 256, lanes at 2,048, vectors per
expert at 1,048,576, symbols per expert at 16,777,216, total symbols at
268,435,456, one logical frame at 64 MiB, and the literal container at 256 MiB.
Malformed input receives `FormatError` before factorial, directory iteration,
slice, conversion, or allocation.

## Format and byte ownership

```text
4096-byte global header
page-aligned external expert directory
page-aligned literal Q16 model
page-aligned expert frame 0
...
page-aligned expert frame E-1
```

Let `G=frames_offset`, and let `F_e` be expert `e`’s padded frame bytes. The
owner share and routed cold amplification are

```text
owner_e = (G + E*F_e) / E
cold_e  = G + F_e
A_e     = E*(G + F_e) / (G + E*F_e).
```

The `<2` decision is strict integer cross-multiplication. The audit’s
`G=8192, F=[4096,8192,...]` counterexample is fixed at exactly `A_0=12/5` and
must fail.

## Authenticated verification

No producer root is embedded here: that would be self-authentication. An
independent auditor computes and publishes the source root out of band.

From a clean exact source directory, first compute the observation without
imports:

```bash
python3 -I -S -B source_bootstrap.py --print-observed-root
```

Then verify against the independently recorded value:

```bash
python3 -I -S -B source_bootstrap.py \
  --expected-root <64-hex-root> \
  --entry verify
```

The verifier has no CPU-only or GPU-skip mode. It requires CuPy, NVML, and the
provided RTX 5090. It records the interpreter hash and all authenticated member
hashes in its stdout receipt. Extra, missing, modified, or symlinked source
members are rejected before import.

The synthetic runner is similarly reachable only through the bootstrap:

```bash
python3 -I -S -B source_bootstrap.py \
  --expected-root <64-hex-root> \
  --entry synthetic -- \
  --output-path /absolute/nonexistent/output
```

Its publication remains explicitly `UNSEALED`; this producer work does not run
or freeze it.

## Files

- `design_lock.json`: v1 contract and caps.
- `silt_v1.py`: bounded CPU reference, format, canonical decoder, and ledgers.
- `independent_decoder_v1.py`: separate parser/decoder/re-encoder and owner
  ledger.
- `cupy_backend_v1.py`: mandatory search and complete measured telemetry.
- `safe_publish.py`: Linux exclusive atomic publication.
- `source_bootstrap.py`: stdlib-only content-addressed bootstrap.
- `verify_source_v1.py`: authenticated mandatory-GPU verifier.
- `test_source_only_v1.py`: hostile suite.
- `run_synthetic_v1.py`: authenticated synthetic-only unsealed runner.
- `POSTIMPLEMENTATION_REVIEW.md`: producer’s non-authoritative review boundary.

