#!/bin/bash
set -u

CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -I -B \
  /tmp/tactic_actual_coarse_n18_v6_runtime_316625/research/tactic_dh384_finite_v3_bf0659d1/dispatcher.py \
  --authorization RUN_REVIEWED_TACTIC_DH384_FINITE_V3_ONE_BOUND_EXPERT \
  --package-manifest-sha256 bf0659d1fd6742768d14790ea980aa17321818d15e19ddd7d0dfaa8a223009b8 \
  --repo-root /tmp/tactic_actual_coarse_n18_v6_runtime_316625 \
  --v6-package-dir /tmp/tactic_actual_coarse_n18_v6_runtime_316625/research/tactic_actual_coarse_n18_v6 \
  --v6-result-dir /workspace/tactic_actual_coarse_n18_v6_qwen_pilot_result_20260902T1530Z_fe4fd2b8 \
  --input-manifest /workspace/tactic_actual_coarse_n18_v6_qwen_pilot_input_20260902T1530Z_fe4fd2b8/input_manifest.json \
  --launch-review /tmp/tactic_dh384_finite_v3_launch_review_704f935a.json \
  --launch-review-sha256 704f935a6f9949f600dff473cdf0e7b54f4700fc95a29c2f5a7273f48e5d505f \
  --output-dir /workspace/tactic_dh384_finite_v3_qwen_result_20260902_bf0659d1 \
  > /tmp/tactic_dh384_finite_v3_qwen_bf0659d1.log 2>&1
RUN_STATUS=$?
printf '%s\n' "$RUN_STATUS" > /tmp/tactic_dh384_finite_v3_qwen_bf0659d1.exit
exit "$RUN_STATUS"
