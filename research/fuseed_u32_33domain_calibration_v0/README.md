# FUSEED-U32 source-free 33-domain calibration v0

This unsealed CuPy calibration emulates the frozen FUSEED stage-0 runtime
shape: one warp generates 256 `normal4` bundles (1,024 values), rounds anchors
through BF16, accumulates common and 33-domain sufficient statistics in FP64,
round-trips fitted affine parameters through FP16, emits
`q[33, 2^24]`, validates every q value, and performs exact tie-aware Top-8192
selection for each domain.

The targets are deterministic synthetic values. The probe has no model/source
argument and does not test producer ABI parity, Qwen capture, controls,
selection validity, journals, global merges, or a physical codec. Linear
three-ABI projections are planning measurements only.

## Full-shard surrogate result and red-team disposition

The corrected script (SHA-256
`6d3cdac6ab4f1a1fcbe742f43a2fd817c8bf56d1bfeff9e38c2432fe6848149a`)
ran two complete `2^24`-candidate shards on the pinned RTX 5090/CuPy 14.2
runtime. The kernel used 106 registers/thread, reported zero bytes of local
storage (no register-spill proxy), and emitted the exact
`33 x 2^24 x 4 = 2,214,592,512`-byte q array. Both repetitions had identical
q sentinel and domain Top-K seed/value hashes.

Median kernel time was `3.3351742995437235 s` per shard. Multiplying it by
three ABIs and 256 shards gives `2561.4138620495796 s`; the analogous
finite-validation/Top-K surrogate is `2992.413043230772 s`.

An independent red-team then found that this kernel is **not runtime-
equivalent to FUSEED-v1**: it calls `curand_init` and `curand_normal4`
separately for every bundle, whereas FUSEED-v1 requires a direct
Philox4x32-10 counter implementation plus exact Box--Muller and expressly
forbids per-bundle CURAND state setup. The end-to-end multiplication also
charges the first cold Top-K cost once per shard although the intended runtime
separates one-time cold compilation/allocation.

Disposition: **BLOCKED_SURROGATE_NOT_A_V1_RUNTIME_DECISION**. These timings
neither pass nor kill FUSEED-v1's prospective 900-second gate. No model payload
was opened. A distinct exact direct-counter calibration is mandatory.

## Attempt log

The first smoke attempt used script SHA-256
`31f5ba6e93c6476aa2671e65f2a6d7f3cd5d89a891ce587a236169f3919d9f76`
and stopped during NVRTC compilation before a kernel launch or output write:
isolated CuPy could not resolve the unnecessary host `math.h` include. The
next attempt removes only that unused include; no runtime or scoring operation
changed.

The second smoke attempt used script SHA-256
`0c33b96fc833ac5e1193de4fd6f50f384d70ed66db20689821f9125db3fde5eb`.
The kernel compiled and executed, but the fail-closed Top-K cardinality check
caught a reversed provisional partition (largest q values although lower q is
better). It stopped before an output write. The next attempt changes only the
partition boundary and slice to select the lowest q values.

## Verification

Run `./verify_result.ps1` in PowerShell to recompute the source binding,
shape/memory arithmetic, replay hashes, medians, and explicitly non-authority
surrogate projections. `artifact_sha256.txt` lists all package file digests
and excludes its own digest.
