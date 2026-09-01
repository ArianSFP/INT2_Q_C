# RunPod Tier-A result

Execution completed on the RTX 5090 RunPod after the frozen CPU preflight and
13/13 unit tests passed.

## Outcome

`HARD_KILL_BOUNDED_INITIALIZER_SET`

- source winner: `hf451_layer_reset`, seed `0`,
  `fp32_then_bfloat16` (frozen ordinal 14);
- candidate-selection source capture: `0.00042320022129282986`;
- untouched-validation raw/corrected capture: `-0.0007686974917568978`;
- whole-expert validation SE: `0.0010746954187926254`;
- corrected optimistic upper bound (`+2 SE`): `0.001380693345828353`;
- metadata-adjusted composite capture requirement: `0.14580061597878702`;
- metadata-adjusted current-result capture requirement: `0.19112644610213347`;
- validation role captures: down `-0.001102220899387385`, up
  `-0.00043566258747129716`;
- all whole-expert folds positive: false;
- conservative cold-read amplification: `1.1694907936197916x` (passes `<2x`).

Thus the locked HF initialization family misses even the easier composite
quality threshold by roughly two orders of magnitude.  Bandwidth is not the
bottleneck for this candidate family.

## Independent verification

The CUDA-free verifier returned `PASS`, independently rehashed all 31 eligible
payloads, opened zero excluded payloads, and confirmed no pinned-panel access.

RunPod artifacts:

- `/workspace/INT2__compression/init_anchor_aux_gate_v1/initialization_anchor_result.json`
  (`sha256:6ef38ff14f69ab02caf0e48ad37e5dbc3dfa9ebe7ba5e663f85080114e0f828d`)
- `/workspace/INT2__compression/init_anchor_aux_gate_v1/verification_receipt.json`
  (`sha256:13aa44df48cefa368222f943b9a097f2252ecd17928edceb7d7b8dd5e53f8007`)
- `/workspace/INT2__compression/init_anchor_aux_gate_v1.log`
  (`sha256:2623c7b9dfc63d10a2415aa6c43da0a43a620ad2268419c1f0399f6da69e474c`)

The negative claim remains deliberately narrow: it rejects the 56 sealed
Tier-A Hugging Face/common-seed hypotheses, not every possible production
initializer or procedural predictor.
