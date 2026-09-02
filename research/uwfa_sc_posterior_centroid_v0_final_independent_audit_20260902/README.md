# UWFA-SC posterior-centroid v0 final independent audit

This is a clean-room, source-only audit of producer manifest
`0ef30253d4d31504fbd8f88b8203cf35bce6c14952e570aace44b7bc089cb713`.
It does not reuse or endorse any earlier posterior audit directory.

Run on a source-only host with NumPy available:

```text
python -I -B research/uwfa_sc_posterior_centroid_v0_final_independent_audit_20260902/independent_audit.py
```

The audit accesses no Qwen/model payload, completed v9 result, BF16 score
panel, Gaussian control, RunPod or CUDA device. See `AUDIT_REPORT.md` for the
scope and `AUDIT_RESULT.json` for the machine-readable verdict.
