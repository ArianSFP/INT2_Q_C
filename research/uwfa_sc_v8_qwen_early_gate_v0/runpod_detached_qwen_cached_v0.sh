#!/bin/sh
set -u
umask 077

result=/workspace/uwfa_sc_v8_qwen_early_gate_v0_result_20260902d
log=/workspace/uwfa_sc_v8_qwen_early_gate_v0_result_20260902d.log
exit_receipt=/workspace/uwfa_sc_v8_qwen_early_gate_v0_result_20260902d.exit

if [ -e "$result" ] || [ -e "$log" ] || [ -e "$exit_receipt" ]; then
    exit 125
fi

/workspace/int2-cupy-venv/bin/python -I -B \
  /workspace/INT2_Q_C_qwen_v8_20260902opt/research/uwfa_sc_v8_qwen_early_gate_v0/early_gate.py \
  --authorization RUN_EXACT_QWEN_EARLY_KILL_NO_CONTROLS_NO_CLAIM_V0 \
  --v8-package /workspace/INT2_Q_C_qwen_v8_20260902opt/research/unifilar_wfa_entropy_census_stage0_v8 \
  --strata-common /workspace/INT2__compression/strata_expert_local_codec/common.py \
  --frozen-auditor /workspace/INT2__compression/strata_v2_klt_mixed_independent_auditor_v1.py \
  --artifact /workspace/INT2__compression/strata_expert_affine_milestone_v1/strata_expert_affine_n20n21.bin \
  --output-dir "$result" >"$log" 2>&1
code=$?

tmp_receipt="${exit_receipt}.tmp.$$"
printf '%s\n' "$code" >"$tmp_receipt"
mv "$tmp_receipt" "$exit_receipt"
exit "$code"
