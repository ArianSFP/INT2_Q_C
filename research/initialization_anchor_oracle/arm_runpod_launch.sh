#!/usr/bin/env bash
set -euo pipefail

workspace=/workspace/INT2__compression
output_dir=${workspace}/init_anchor_aux_gate_v1
log_path=${workspace}/init_anchor_aux_gate_v1.log
release_path=/tmp/init_anchor_tier_a_release_v1

if [[ -e "${output_dir}" || -e "${log_path}" || -e "${release_path}" ]]; then
  echo "refusing launch: output, log, or release sentinel already exists" >&2
  exit 70
fi

( set -o noclobber; : > "${log_path}" )
nohup bash -c '
  set -euo pipefail
  while [[ ! -e /tmp/init_anchor_tier_a_release_v1 ]]; do
    sleep 0.2
  done
  cd /workspace/INT2__compression
  exec env PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    /workspace/int2-cupy-venv/bin/python \
    INT2_Q_C/research/initialization_anchor_oracle/initialization_anchor_gate.py \
    --workspace-root /workspace/INT2__compression \
    --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
    --output-dir /workspace/INT2__compression/init_anchor_aux_gate_v1 \
    --backend cupy
' >> "${log_path}" 2>&1 < /dev/null &

pid=$!
printf 'PID=%s\nRELEASE=%s\nLOG=%s\n' "${pid}" "${release_path}" "${log_path}"
