#!/bin/bash
set -u

CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -I -B \
  /tmp/tactic_cage_graph_krylov_oracle_v0_run_4d5220fe_clean/run_oracle.py \
  --authorization RUN_TACTIC_CAGE_GRAPH_KRYLOV_ORACLE_V0_QWEN_PILOT \
  --package-manifest-sha256 4d5220fe36a9ad5ca30579898a714338547fe2f64ca38be58821dab03fb173e8 \
  --v6-package /tmp/tactic_actual_coarse_n18_v6_runtime_316625/research/tactic_actual_coarse_n18_v6 \
  --v6-package-manifest-sha256 31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d \
  --v6-predecessor-lock-sha256 645310404673e944c0f61e08747b4d7d50e6681cd450eb829acd8614c41f4322 \
  --v6-runtime-lock-sha256 de1464d23de161d90f0784183743252631385ad69ba2620697dea7df763c3490 \
  --v6-result-dir /workspace/tactic_actual_coarse_n18_v6_qwen_pilot_result_20260902T1530Z_fe4fd2b8 \
  --v6-complete-sha256 6b5e96c42518a29493e68237d649daad2e25f44a509ce7535425f83fd79fbb37 \
  --input-manifest /workspace/tactic_actual_coarse_n18_v6_qwen_pilot_input_20260902T1530Z_fe4fd2b8/input_manifest.json \
  --input-manifest-sha256 6f6a0f174cd5b9c2b52ef29efd612e4520ef77afa6cc950ebec8c7e055fedcaa \
  --output-dir /workspace/tactic_cage_graph_krylov_oracle_v0_qwen_result_20260902_4d5220fe_retry1 \
  > /tmp/tactic_cage_graph_krylov_oracle_v0_qwen_4d5220fe_retry1.log 2>&1
RUN_STATUS=$?
printf '%s\n' "$RUN_STATUS" > /tmp/tactic_cage_graph_krylov_oracle_v0_qwen_4d5220fe_retry1.exit
exit "$RUN_STATUS"
