# RAVEL decoded-residual LUT stage-0

Status: **source candidate; not yet executed.**

RAVEL (**R**esidual-**A**ware **V**alue/**E**dge **L**ookup) tests a nonlinear,
decoder-visible correction that is not contained by the completed scale/bias/
affine envelope.  It never predicts weights from another expert.  Instead it
uses only an already decoded STRATA weight, its row RMS, and its two decoded
horizontal neighbours to address a tiny correction table.

For every reconstructed coordinate, the deterministic key is

```text
(role, row-RMS class, normalized-amplitude bin, left edge state, right edge state)
```

with `3 * 4 * 32 * 4 * 4 = 6,144` entries.  The edge state combines neighbour
sign and whether the neighbour magnitude exceeds the centre magnitude.  All
features are computed from the decoded matrix; no source label or per-weight
side stream is legal.

The fixed split is whole-expert `fit={0,2,3,5}`, `holdout={1,4}` on the pinned
six-expert panel.  The emitted diagnostic table is learned on fit experts,
rounded to canonical little-endian FP16, padded with its schema to exactly
16,384 bytes, and then replayed on holdout experts.

The hard-kill envelope is deliberately stronger.  It refits every table entry
in exact FP64 directly on the complete holdout residual and charges only the
same nominal 16-KiB packet.  This leaks the answer and does not emit a codec,
but it is the exact least-squares projection onto the frozen lookup cells.
Every realizable shared RAVEL table lies inside that envelope.

The side rate over the six-panel denominator is

```text
8 * 16,384 / 28,311,552 = 0.004629629629629629 bpw.
```

The gate also grants favorable transfer of the observed correction fraction
after removing that rate from the coarse payload.  With a held-out baseline
factor `F0`, the envelope is

```text
F_favorable = F0 * (SSE_corrected / SSE_baseline) * 2**(2 * side_bpw).
```

If the source-leaking envelope remains above `F=0.8`, the cell is killed before
matched controls, a lower-rate coarse encode, or table-kernel engineering.  A
survivor is only permission for those next steps, never a target result.

The table is four pages.  Conservatively adding all four pages to the audited
worst STRATA cold read keeps the architecture far below `2x`; no warm-cache
assumption is used.

Run only after source verification and GPU coordination:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  /workspace/int2-cupy-venv/bin/python -B -I \
  research/ravel_decoded_residual_lut_stage0_v0/ravel_stage0.py \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --output /workspace/ravel_decoded_residual_lut_stage0_v0_run \
  --authorization OPEN_AUTHENTICATED_DECODED_PANEL_FOR_RAVEL_STAGE0_V0
```

The output must be absent.  It is external to this source package and contains
`fit_table_packet.bin` plus `result.json`.

Claim boundary: this is an architecture-scoped raw-MSE test of one 6,144-cell
decoded lookup.  It is not a converse for arbitrary neural correction, not an
activation-aware result, and not a finite result at a reduced coarse rate.

