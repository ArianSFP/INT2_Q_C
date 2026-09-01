# PMG1 tetrad auxiliary stage-1 screen v0

Status: **development-panel falsification only**. This package neither
authorizes nor reports a validation, pinned-panel, compression, or target-pass
result.

This screen answers the cheapest useful question about the four PMG seeds
selected by the historical stage-0 search: after fitting the four anchors on
the already-frozen stage-1 fit coordinates, do decoded-FP16 affine
coefficients retain their gain on the disjoint stage-1 score coordinates?

The experiment opens only the 23 historical layer-15 Up/Down selection
matrices. It reconstructs the prospective stage-1 coordinate sets from the
frozen PMG plan, generates anchor coordinates with a CuPy/NVRTC kernel using
the exact direct Philox/cuRAND Box-Muller/BF16 ABI, fits each expert-role
matrix independently, rounds all four coefficients and the intercept through
IEEE binary16, and scores only the held-out coordinates. Sixteen deterministic
within-matrix coordinate-scramble controls quantify chance fitting.

The four fixed seed labels are:

```text
3306464084, 235286348, 2174751347, 256779041
```

The conditional planning capture `0.1457530997916614` is a prioritization
threshold, not a converse: missing it is labelled
`POLICY_REJECT_INCONCLUSIVE`, never a universal family hard kill. A survivor
still says nothing about Gate, rebuilt residual role/polar geometry, a finite
codec, fresh Qwen experts, or the final same-rate `F <= 0.8` objective.

Run on the supplied RunPod with an absent output directory:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B \
  stage1_screen.py \
  --workspace /workspace/INT2__compression \
  --manifest /workspace/INT2__compression/agent_rd_structure_diag_cross_expert_sources.json \
  --output /var/tmp/pmg1_tetrad_aux_stage1_v0_result
```

Promotion to a larger stage-2 screen is permitted only when the raw aggregate
source-energy capture clears the frozen planning threshold and every role has
positive capture. Otherwise the script stops the tuple at this cheap gate.

