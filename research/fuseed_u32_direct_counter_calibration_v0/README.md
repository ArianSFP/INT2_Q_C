# FUSEED-U32 exact direct-counter calibration v0

This source-free CuPy package repairs the decisive mismatch in the earlier
surrogate calibration. Its performance kernel never calls `curand_init` or
`curand_normal4`: it builds the generic 128-bit counter, executes the pinned
Philox4x32-10 device function directly, calls the pinned cuRAND device-inline
Box--Muller for each output pair, applies exact role scale bits, and rounds
through BF16 before the 33-domain FP64/FP16-affine metric.

Before timing, a separate reference kernel compares raw float32 bits, scaled
BF16 bits, and terminal counters against `curand_init`/`curand_normal4` across
seed carries, sequence/lane boundaries, counter carries, offsets, and call-size
endpoints. The performance run emits `q[33,2^24]`, validates it, and computes
exact implicit-seed Top-8192 three times.

This remains a source-free calibration, not a Qwen or retention result. It
does not bind the final frozen 256-bundle plan or journal/global-merge path;
those require an independent audit before any payload authorization.

## Measured result

The script (SHA-256
`f5a7c8b9a525e02d469ca974f9a6607030b2ca2822b66d4bce31604251516ed5`)
passed 132 direct/reference vectors. Raw normal float32 bits, role-scaled BF16
bits, and terminal counters were identical; the Philox zero-key/counter KAT
was `6627e8d5 e169c58d bc57ac4c 9b00dbd8`. The vector, raw, scaled, and terminal
hashes are respectively `3bcb66a0...`, `58601f45...`, `3dfafb5b...`, and
`0b539764...` in `result.json`.

Three complete `2^24`-candidate repetitions emitted the exact
`33 x 2^24` q array and identical q/Top-K hashes. The RTX 5090 kernel used 108
registers/thread and zero reported local-storage bytes. Median kernel time was
`3.3335224930197 s` per shard. Across three ABIs and 256 shards/ABI, kernel
time alone projects to `2560.1452746391296 s`; warm finite-validation plus
Top-K projects to `2672.980880373507 s` after charging the measured cold
selection excess only once.

FUSEED-v1 prospectively capped this calibration at 900 seconds. Both valid
direct-counter projections exceed that cap, so the provisional decision is
**EARLY_KILL_RUNTIME_NO_QWEN_PENDING_INDEPENDENT_CALIBRATION_AUDIT**. The gate
is not relaxed after observing the result, and no Qwen payload was opened.

## Verification

`./verify_result.ps1` recomputes package/source bindings, header/runtime
records, direct-call counts, parity hashes, q arithmetic, repeat determinism,
timing medians, the one-time cold correction, and the 900-second predicate.
The artifact digest list excludes its own hash.
