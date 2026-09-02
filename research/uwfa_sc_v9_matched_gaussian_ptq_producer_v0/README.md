# UWFA-SC v9 full-PTQ matched-Gaussian producer v0

## Outcome

An honest full re-encode is mechanically feasible with committed production
primitives. No new quantizer or polar encoder is required.

The exact path is:

1. generate and retain eighteen BF16 matrices for each frozen null seed;
2. call the unchanged `strata_v2_codec.emit_and_lock.build_staging` primitive,
   which performs the current Q15 XKLT, FP32 rounding, BF16-RNE staging,
   equipopulous STRATA labeling and deterministic grouping;
3. regroup those exact staged words with the unchanged current
   `strata_expert_local_codec` N20/N21 ownership map;
4. call the unchanged `strata_v2_codec/polar_encoder.py` once for every sealed
   block through `strata_expert_local_codec.run_and_pack`;
5. emit the literal 8,847,360-byte `PLRLOC3` artifact;
6. independently decode every block, canonically re-encode every arithmetic
   payload, score against the retained BF16 control source, and require its
   reconstruction digest to equal the sealed v8 adapter's digest.

This package is source-only and nonpromoting. It did not access RunPod, CUDA,
Qwen/model payloads, a live v9 result, a Gaussian payload or a current STRATA
artifact. Direct execution exits 3. A reviewed dispatcher must authenticate
and inject the exact runtime modules and external receipts.

## Frozen null law

The eight global seeds remain:

```text
10619863, 10619881, 10619909, 10619927,
10619953, 10619971, 10619999, 10620017
```

An independent source-moment auditor supplies one sealed row per canonical
slot and SwiGLU role. Each row binds the original BF16 matrix hash, shape,
binary64 mean, centered SSE and energy. The generator key contains only:

```text
(global seed, canonical slot, role, shape, mean, centered SSE)
```

It contains no checkpoint, model, layer or expert identity. PCG64 produces an
independent binary64 normal vector. A fixed six-iteration affine correction
followed by exact FP32-to-BF16 round-to-nearest-even minimizes BF16 moment
bias. The selected iteration must satisfy both preregistered tolerances:

```text
|mean_control-mean_source| / RMS_source <= 2^-17
|centered_RMS_control/centered_RMS_source - 1| <= 2^-15
```

All generated BF16 bytes are retained. A digest-only receipt is insufficient:
the later controls auditor must independently reopen the retained source
container, recompute moments, rerun the source generator from its authenticated
capsule, and reproduce every matrix byte.

## Source-independent geometry invariant

`universal_format_geometry()` is frozen before control generation. It binds:

- six canonical panel slots;
- Gate, Up and Down roles and their storage shapes;
- hidden/intermediate dimensions and total weight count;
- fifteen block ordinals, log2 sizes and routed owner slots.

It deliberately excludes source-derived labels, profiles, selected-symbol
counts, role-contribution counts and arithmetic lengths. An honest control
reruns the full PTQ algorithm, so those values are allowed to differ and each
control authenticates its own complete geometry.

The numeric route required by the current file format uses `(namespace=0,
slot=0..5)` only. This is a format address, not model identity. The scientific
generator and probability model see canonical slots, roles and shapes.

## Explicit v8 consumer incompatibility

The sealed v8 `controls_phase` cannot consume these honest artifacts unchanged:

1. it requires `control_structural_geometry == source_structural_geometry`,
   but that digest includes independently derived profiles, symbol counts and
   label-dependent role contribution maps;
2. its six-field bundle has no retained generated BF16 source container, so an
   independent moment replayer cannot recompute the source moments or prove
   that the artifact was encoded from those bytes;
3. the historical independent scorer requires Qwen tensor-name strings even
   though its numerical decode is generic.

These are interface failures, not permission to copy source metadata, mutate
SC labels, or weaken the null. `BLOCK.json` requires a separately reviewed v9
controls bridge that:

- authenticates the pre-frozen universal format geometry;
- authenticates every control's own source bytes, full geometry, score and
  reconstruction;
- uses the same quantizer/polar source snapshot as the Qwen primary;
- independently repeats all 150 WFA cells for every control;
- never reuses the Qwen-selected cell;
- requires all eight nulls for a positive specificity result.

## Runtime closure

For each seed the producer emits:

- `SOURCE_PANEL.json` plus all eighteen BF16 source files;
- the exact transform staging and sealed current plan;
- all fifteen encoder metadata records and literal current artifact;
- an independent decode/score record;
- a v8-compatible score receipt;
- a BF16 moment-replay receipt;
- a v9 control binding containing both source-specific and universal closure;
- `COMPLETE.json`, written last.

The eight-control root hashes all members and is completed only after all eight
source panels and artifacts exist. It explicitly records that all-150 WFA
search has not yet run; this producer cannot itself promote the result.

The root also stores the literal pre-frozen universal format geometry. Every
control score stores its own recomputable full and structural geometry, while
the v9 binding requires the complete named NumPy/CuPy, polar-tree, transform,
quantizer, packer, independent-decoder and adapter source closure. An empty or
partially named "same pipeline" digest is rejected.

## Source-only verification

From the repository root, with an available Python/NumPy environment:

```text
python -I -B research/uwfa_sc_v9_matched_gaussian_ptq_producer_v0/test_source_only.py
python -I -B research/uwfa_sc_v9_matched_gaussian_ptq_producer_v0/verify_source.py \
  --package /absolute/path/to/research/uwfa_sc_v9_matched_gaussian_ptq_producer_v0
```

No payload launch is authorized until a new independent source audit approves
both this producer and the separate v9 consumer bridge.
