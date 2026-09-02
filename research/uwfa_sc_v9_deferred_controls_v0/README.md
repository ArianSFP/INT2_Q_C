# UWFA-SC v9 deferred controls v0 — source-only bounded design

This immutable sibling freezes the deferred control experiment for the exact
UWFA-SC v9 primary aperture. It did not access RunPod, the live primary result,
Qwen/model payloads, original BF16 sources, Gaussian controls, NumPy, CuPy or
CUDA during construction.

## Outcome

The selected-symbol shuffle path is decoder-closed and specified. The matched
Gaussian path is **not** decoder-closed by the sealed v8 APIs, so v0 is a
bounded `BLOCK`, not an executable payload gate.

The key distinction is important. `stage0_census.controls_phase` can consume
eight already-authenticated Gaussian control artifacts and rerun all 150 WFA
cells. `StrataSCAdapter` can decode/extract an existing artifact. Neither API
constructs Gaussian BF16 SwiGLU weights and then runs the identical current
STRATA transform, polar quantizer and artifact encoder. Mutating decoded SC
bits would test a label shuffle, not a Gaussian PTQ control. The latter is
forbidden here.

An independently audited primary survivor is also not yet pinned into this
source tree. A raw `RESULT.json`, a producer-authored `COMPLETE.json`, or a live
directory is not authority. Therefore `deferred_controls.py` fails before
dynamic path handling and returns the static block record.

## Frozen experiment after the blockers are resolved

The successor must retain the exact v8 bank, component folds, Q0.16 fitting,
literal serializer/page/model ledger and deterministic ties. It may not reuse
the Qwen-selected cell on any control.

The eight matched controls use the already frozen v8 seed order:

```text
10619863, 10619881, 10619909, 10619927,
10619953, 10619971, 10619999, 10620017
```

All eight bundles and moment replays authenticate before the first numeric
fit. Controls are then fit in seed order. One null with absolute physical
saving greater than or equal to Qwen is sufficient to stop and reject source
specificity. A survivor must complete all eight.

Each complete pipeline is the exact v9 aperture:

```text
15 streams
126,627,266 selected symbols
150 candidate cells
3 disjoint owner-component folds plus final fit/score
38,621,316,130 cell-symbol updates
```

Thus one decisive null costs 38,621,316,130 updates; all eight cost
308,970,529,040. Early stopping saves failure work but cannot reduce the
positive-pass requirement.

## Structure-destroying diagnostics

Six diagnostic variants are frozen. Every one repeats all-150 selection and
the exact literal component score:

1. the retained v8 within-public-context, phase-preserving permutation;
2. a cross-stream permutation within literal role, STRATA `profile_q`, polar
   level, prior bin and phase;
3. the same bucket without phase, explicitly destroying phase dependence;
4. the retained v8 triplet chunk shuffles at 32, 128 and 512 symbols.

The two new transformations use a seeded, bucket-specific bijective affine
gather. `control_core.py` is the standard-library reference. A future CuPy
implementation must byte-match it on KATs before any payload launch. The six
diagnostics cost at most 231,727,896,780 additional updates.

Gate/Up/Down “role” is used only when it is literally present in the extracted
stream metadata. A block whose charged contributions span roles remains
`mixed`; it is never relabeled. `profile_q` is the only literal STRATA field
used as the stratum.

Matrix row/column shuffling is deliberately not faked at this aperture. A
selected SC decision is not a matrix coordinate after the RHT and multilevel
polar decode. A true row/column control must permute source weights and rerun
the complete current PTQ encoder, which is the same missing producer boundary
as the Gaussian controls.

These shuffles diagnose where a saving comes from. They do not rescue a failed
physical codec or by themselves promote a result.

## CuPy execution disposition

The future runner reuses the sealed v8 CuPy count/length backend. For the two
new shuffles it forms bucket IDs and applies the affine gather on GPU, releases
all sort/group scratch, then builds the normal v8 backend cache. Only one CUDA
owner process is allowed. Each matched control gets a fresh backend. No CUDA
work was run for v0.

## Source-only verification

From the repository root:

```text
python -I -B research/uwfa_sc_v9_deferred_controls_v0/test_source_only.py
python -I -B research/uwfa_sc_v9_deferred_controls_v0/verify_source.py \
  --package /absolute/path/to/research/uwfa_sc_v9_deferred_controls_v0
```

Direct execution is intentionally inert and exits 3:

```text
python -I -B research/uwfa_sc_v9_deferred_controls_v0/deferred_controls.py
```

The next executable version must be a new `v1` sibling with the independent
primary-audit manifest/receipt and matched-control producer/bundle roots fixed
as literal digests. This v0 must not be edited in place.

