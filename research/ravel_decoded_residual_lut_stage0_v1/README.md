# RAVEL decoded-residual LUT stage-0 v1

Status: **sealed source-only candidate; not executed.**

RAVEL-v1 is the repaired preregistered stage-0 test of one decoder-visible,
shared 6,144-entry residual table. It preserves the six-expert whole-expert
split and favorable-transfer gate while correcting the invalid v0 projection,
hardening input authentication, and defining a finite packet/result protocol.

No model payload, decoded panel, fresh validation data, CuPy runtime, GPU, or
network resource was opened while authoring or sealing this package.

## Frozen lookup and split

Every coordinate uses only its decoded reconstruction and fixed role metadata.
The flattening order is

```text
((((role*4 + row_class)*32 + amplitude)*4 + left_state)*4 + right_state)
```

with right state fastest and role order `(gate, up, down)`. One table is shared
by every role and all six panel experts. Fit experts are `{0,2,3,5}` and
holdouts are `{1,4}`.

Feature semantics are part of the versioned packet:

- row scale is `max(sqrt(mean(decoded_row**2)), 1e-30)` in FP64;
- row class counts strict comparisons of `log2(row_scale/matrix_scale)` against
  `(-0.25, 0, 0.25)`; equality stays in the lower class;
- amplitude is `floor((decoded/row_scale + 4)*4)` clipped to `[0,31]`, with
  lower edges inclusive and upper edges exclusive before saturation;
- edges are noncyclic self-clamps: a missing neighbor equals the center;
- edge state is `2*(neighbor >= 0) + 1*(abs(neighbor) > abs(center))`; zero is
  nonnegative and magnitude ties are false.

The finite FP16 table consumes residuals only from fit experts. The favorable
oracle deliberately consumes holdout residuals and is labeled source-leaky.

## Correct raw-MSE projection

For each cell, v1 accumulates

```text
numerator   = sum(row_scale * residual)
denominator = sum(row_scale**2)
table       = numerator / denominator
```

in FP64, with empty cells set to zero. This is the scalar least-squares solution
for the actual correction `table[cell] * row_scale`. The runtime calls it a
*numerical FP64 least-squares projection*, not a bitwise exact reduction, and
requires its replayed SSE to be no worse—within an explicit numerical
tolerance—than both zero correction and the compared legal fit FP16 table.

If the source-leaking favorable oracle remains above `F=0.8`, the exact frozen
cell is killed before matched controls. A survivor is only permission to run
controls and a finite reduced-coarse-rate re-encode; it is not target evidence.

## Rate and read contract

Exactly one 16,384-byte table is charged over 28,311,552 panel weights:

```text
side_bpw = 8*16384/28311552 = 0.004629629629629629
base_payload_cap = 2.5-side_bpw = 2.4953703703703702
```

Four 4-KiB pages add `0.011111111111111112` to the published worst cold-read
amplification `1.1694444444444445`, yielding `1.1805555555555556 < 2`.
Any per-role, per-expert, or multiple-table variant invalidates this ledger and
requires a new physical charge.

## Authenticated inputs

Before parsing an input, the runner rejects symlink/reparse components, opens a
regular file with no-follow semantics, compares identity before/open/after,
reads an immutable byte snapshot, authenticates that snapshot, and parses the
same bytes. This applies to the plan, header, decoded reconstruction, and all 18
BF16 source matrices. Source package hashes are also checked before payload
access and carried into the result.

## Packet and output protocol

`fit_table_packet.bin` is exactly 16,384 bytes:

- bytes `0..4095`: canonical ASCII JSON plus LF and required zero padding;
- bytes `4096..16383`: 6,144 finite little-endian FP16 entries;
- the header binds version, dimensions, one-table count, full feature semantics,
  table offset/length, semantics hash, and table hash.

The runner builds and parses the packet before using its rounded FP16 values
for the finite holdout evaluation. `verify_result.py` contains a separately
implemented parser. Nonfinite pre/post-FP16 values, nonzero padding, schema
drift, and hash mismatch are fatal.

After authorization and device checks, the absent output directory is reserved
before payload access. Files are staged and atomically moved into it; a hashed
`COMPLETE.json` is published last. A directory without that marker or with any
extra/missing member is an incomplete result and must not be consumed. Runtime
failures intentionally leave such a recognizable incomplete directory rather
than overwriting or masquerading as success.

The result emits all 18 per-matrix baseline SSE/energy rows, six holdout finite
and oracle rows, pooled panel/holdout sums, 18 source receipts, packet/source
bindings, dominance checks, rate/read arithmetic, and a canonical result lock.
`verify_result.py` checks the completed three-file result without opening model
payloads.

## Source verification and tests

From the directory directly containing `research/`:

```bash
INT2_PROJECT_ROOT=/workspace/INT2__compression/INT2_Q_C
/usr/bin/python3.12 -B -I \
  "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1/verify_source.py" \
  --package "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1"
cd "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1"
/usr/bin/python3.12 -B -I test_source_only.py
```

## Authorized RunPod launch

The output path must not exist. No launch has been performed by this package:

```bash
INT2_PROJECT_ROOT=/workspace/INT2__compression/INT2_Q_C
env CUDA_VISIBLE_DEVICES=0 \
  /workspace/int2-cupy-venv/bin/python -B -I \
  "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1/ravel_stage0.py" \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --output /workspace/ravel_decoded_residual_lut_stage0_v1_run \
  --authorization OPEN_AUTHENTICATED_DECODED_PANEL_FOR_RAVEL_STAGE0_V1
```

After download, verify only a completed result:

```bash
/usr/bin/python3.12 -B -I \
  "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1/verify_result.py" \
  --source-package "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1" \
  --result-dir /path/to/ravel_decoded_residual_lut_stage0_v1_run
```

Claim boundary: this is an architecture-scoped raw-MSE test of one frozen
RAVEL-6144-v1 table. It is not a converse for arbitrary neural correction, not
activation-aware evidence, and not an achieved codec at a reduced coarse rate.
