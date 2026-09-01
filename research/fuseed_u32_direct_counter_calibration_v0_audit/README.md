# FUSEED-U32 direct-counter calibration v0 independent audit

This package independently authenticates and reviews the frozen, source-free
direct-counter calibration in
`../fuseed_u32_direct_counter_calibration_v0/` against the prospectively frozen
FUSEED-U32-v1 runtime and numerical gates.

The calibration bytes, its own verifier, the direct-counter arithmetic, the
132 recorded parity rows, three deterministic score/Top-K replays, and the
runtime projection arithmetic all authenticate.  The measured kernel-only
projection is `2560.1452746391296 s`, and the warm projection is
`2672.980880373507 s`, both above the frozen `900 s` limit.

The independent release verdict is nevertheless **BLOCK**.  The calibration
does not meet its parent design's exact preconditions:

- no CUDA driver, NVRTC/compiler implementation, compiler binary/library, or
  compiled PTX/cubin hash is recorded;
- the result does not explicitly receipt the compilation options;
- parity is executed once, while the frozen design requires three new parity
  repetitions;
- only `curand_init(seed, sequence, O + 4*j)` followed by one
  `curand_normal4` is compared; the separately required
  `curand_init(seed, sequence, O)` followed by `j+1` calls is absent;
- Torch initial-seed and terminal generator-state parity is absent; and
- the exact final bundle-plan digest and journal/global-merge performance path
  are expressly omitted.

These findings do not invalidate the observed timing.  They mean only that it
cannot finalize the frozen v1 runtime-kill claim.  A hardened replay must be a
distinct successor and must retain the original 900-second threshold.

`run_source_audit.ps1` performs the independent checks using only PowerShell
and the frozen source/result metadata.  It does not import CuPy, initialize
CUDA, open model data, or use the network.  `verify_audit.ps1` authenticates
this audit package and its internal receipt seal.
