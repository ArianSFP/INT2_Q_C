# Independent source audit — same-layer common-latent gate v0

This package is a payload-blind hostile review of exactly:

- source manifest SHA-256
  `238ce67f3670566c277d7baac019a69227ccff05ae944f1561ff8a0d32b1bce9`;
- source root SHA-256
  `16cceafcfe06e1c2683c0e89048700edd47fda395a2a6d06a70cef19d8eb858b`.

The review opened no Qwen payload and grants no deployment authority. It was
performed by a separate agent and did not edit the producer package.

## Verdict

`BLOCKED_MATERIAL_READ_GATE_DEFECT`

The quantizer, modal latent, entropy identities, fixed-width count descriptors,
threshold derivation, panel bindings, scramble controls, HOLD ordering, and both
read-amplification formulae pass independent checks. CPU/CuPy parity also passed
on the local RTX 3060 with CuPy 14.2.0.

The source cannot yet be deployed because `run_authorized_panel` computes all
four physical-envelope results but does not consult their `status`,
`capacity_ok`, or `strictly_below_2x` values when setting the final status or
`eligible_for_finite_coder_research`. The audit contains a source-free,
Qwen-shaped regression in which all four envelopes fail while the function
returns `SURVIVE_IDEAL_APERTURE_REQUIRES_FINITE_CODER` and `eligible=true`.

Required repair: freeze an explicit feasible-rate selection rule and make any
promoting/eligible status require the selected envelope to have both
`capacity_ok=true` and `strictly_below_2x=true`. Add the regression to the
producer's mandatory source tests, refreeze, and obtain a fresh independent
review before payload authorization.

## Reproduction

The audit reads only the frozen source package and an already recorded JSON
result used to corroborate the 32 Qwen path/hash bindings. It does not accept a
payload-root argument.

```powershell
$audit = "C:\INT2__compression\INT2_Q_C\research\same_layer_common_latent_entropy_gate_v0_independent_source_audit_20260903"
$source = "C:\INT2__compression\INT2_Q_C\research\same_layer_common_latent_entropy_gate_v0"
$prior = "C:\INT2__compression\INT2_Q_C\research\same_layer_expert_alignment_superoracle_v0_runpod_result_20260901\result.json"
& "C:\INT2__compression\.venv-cupy\Scripts\python.exe" -B "$audit\audit_source.py" --source-package $source --prior-binding-result $prior
```

`AUDIT_RECEIPT.json` is the canonical output from that command.
`CUPY_PARITY_RECEIPT.json` records the independently observed source-free GPU
preflight. `verify_audit.py` authenticates this review package against an
externally supplied manifest digest.
