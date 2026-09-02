# SILT-INT2 source-free finite-mechanism prototype v0

Status: **source ready for independent audit; no result or manifest is frozen**.

This directory implements a sealed-by-construction *mechanism boundary* for the
surviving SILT idea.  It deliberately accepts no Qwen, model-weight, current
codec, or matched-control payload.  All source arrays are generated internally.
A synthetic PASS proves only that the finite transform, physical stream,
independent decoder, byte ledger, cold-read ledger, and GPU search behave as
specified.  It is not a source-gain or model-quality result.

## Mechanism in one paragraph

For each positive-width lane vector, a canonical balanced binary tree applies
exact triangular lifting in either GF(2) or Z4.  Each internal node stores one
three-bit selector `(swap, p, u)` and maps a pair `(x,y)` to

```text
d = (y - p*x) mod A
c = (x + u*d) mod A,       A in {2,4}.
```

The inverse is exact:

```text
x = (c - u*d) mod A
y = (d + p*x) mod A.
```

Odd lanes are carried unchanged, so every positive lane count is legal and a
width `L` tree always emits exactly `L-1` details and one root.  A bounded
64-state, Q16-normalized unifilar model codes roots and details using a finite
32-bit arithmetic reference stream.  The model, factoradic permutation, tree
selectors, frame headers, tails, CRCs, page padding, and every byte fetched for
one expert are charged.

## Files

- `design_lock.json` — format, source boundary, kill rules, and compute contract.
- `silt_mechanism.py` — CPU bit-exact producer/reference, finite coder,
  container, and ledgers.
- `independent_decoder.py` — independent parser/decoder/re-encoder; it does not
  import the producer.
- `cupy_backend.py` — mandatory optimization/search backend and measured GPU
  telemetry.
- `test_source_only.py` — hostile/adversarial source-only suite.
- `verify_source.py` — no-input test entry point.
- `run_synthetic.py` — canonical, synthetic-only, deliberately **unsealed**
  mechanism replay.

There is intentionally no result manifest in this directory.  The canonical
runner refuses to overwrite an output directory and labels its output
`UNSEALED_*`.

## Canonical metadata and charged bytes

The lane permutation is encoded as its Lehmer/factoradic rank using the minimum
whole-byte width needed for `L!` states.  The tree has exactly `L-1` nodes and
its selectors cost exactly `ceil(3(L-1)/8)` bytes; unused selector bits must be
zero.  No permutation, tree, model, header, page tail, or common read is free.

The literal physical layout is:

```text
4096-byte global header and expert directory
page-aligned serialized Q16 model
page-aligned expert frame 0
page-aligned expert frame 1
...
```

Each frame contains a 256-byte header, factoradic permutation, packed selectors,
finite arithmetic bytes, and zero page tail.  Header and body CRCs are checked;
the global header also authenticates the exact model bytes with SHA-256.  Frame
offsets must be contiguous and page aligned.  Any nonzero unused bit or byte is
rejected.

For `E` experts, global bytes `G`, local frame bytes `F_e`, and total bytes `T`,
the reported cold-read amplification is

```text
A_cold(e) = (G + F_e) / (T / E).
```

It therefore charges the common header/model on every cold expert dispatch.
The source-free early gate requires `max_e A_cold(e) < 2`.

## Bounded Q16 state

The detail stream resets publicly every 32 symbols.  GF(2) uses six residue
checks and Z4 uses three, fitting the running modular accumulators in exactly 64
states in both cases.  Context and successor functions are decoder-visible and
integer-only.  Every probability row contains strictly positive `uint16`
frequencies summing to exactly 65,536.  The serialized table is literal; its
bytes and page rounding are charged.

The current entropy stream is a deterministic, general-alphabet 32-bit
arithmetic reference coder, not a production rANS claim.  It is sufficient to
make the test finite and independently reproducible.  A later backend may
replace it only through a new format version and a fresh byte-identical audit.

## Mandatory CuPy optimization path

CPU is the reference and independent decode path.  Metadata optimization is
not permitted to fall back to CPU.  `search_metadata_cupy` fails closed when
CuPy/CUDA is unavailable, runs every candidate transform on the GPU, and ranks
candidates by the exact held-out finite arithmetic codelength after fitting on
a disjoint synthetic training split.  The selected train and validation
coefficients must equal CPU byte-for-byte, and the selected GPU inverse must
equal the original leaves.

The canonical replay requires the device name to contain `5090`.  It records,
without inference:

- CUDA-event H2D milliseconds;
- CUDA-event kernel milliseconds;
- CUDA-event D2H milliseconds;
- host `perf_counter` wall milliseconds;
- sampled device-used bytes, with the measurement method and sample count;
- device name, compute capability, CuPy version, and CUDA runtime version.

NVML is polled when available.  Otherwise synchronized `memGetInfo` samples are
explicitly labeled as phase-boundary samples, not a continuous high-water mark.

## Test and replay

The mandatory RunPod verification command is:

```bash
python3 verify_source.py --require-gpu --require-rtx-5090
```

The source was exercised on the provided NVIDIA GeForce RTX 5090 with CuPy
14.2.0.  The hostile suite covers both alphabets, all eight selector values,
lane counts `1, 2, 3, 5, 17, 97, 257`, odd carries, factoradic extremes,
canonical zero tails, Q16 normalization, arithmetic roundtrip, independent
decode and byte re-encode, truncation, CRC corruption, forged-valid CRCs around
bad selector/rank metadata, matched uniform marginals, physical bytes, and cold
reads.

The canonical synthetic replay is intentionally separate and writes an
unsealed result into a new directory:

```bash
python3 run_synthetic.py --output-dir /new/path/that/does/not/exist
```

It uses disjoint search-train, search-validation, model-fit, and evaluation
seeds.  It runs long-range modular-check sources and iid uniform controls with
the same alphabet, geometry, roots, transforms, and empirical marginal target.

## Ruthless early-kill rules

For each alphabet the synthetic mechanism is killed if any of these fail:

1. physical structured-minus-control saving is strictly greater than 0.15 bits
   per *synthetic leaf symbol*;
2. control rate is at least `0.98*log2(A)` bits per symbol;
3. charged maximum cold amplification is below 2;
4. selected CPU/CuPy coefficients and inverse leaves are exact;
5. the independent decoder reproduces expected leaf digests and the complete
   container byte-for-byte.

The threshold is a mechanism stress test on a constructed dependency.  Passing
it does not imply that the dependency exists in SwiGLU-MoE weights.  Conversely,
failing it kills this finite mechanism before any payload access is considered.

## Universality boundary

Nothing in the format depends on Qwen ancestry, expert identity, a base model,
or a model-specific public reference.  The only source-facing objects in a
future authorized experiment would be finite GF(2)/Z4 label matrices and
decoder-visible geometry.  Gate/Up/Down roles may choose separate public model
IDs, but the transform and decoder grammar are role-agnostic.  Thus the
architecture is eligible for a universal SwiGLU-MoE test; this source-only
prototype does not establish that it will win one.

## Independent-audit handoff

An auditor should copy only this directory into a clean environment, inspect
that no executable accepts an input payload, run the GPU-required verifier, and
then inspect the canonical runner before authorizing any synthetic result.
Source should be hashed/frozen by that independent audit—not by the producer.

