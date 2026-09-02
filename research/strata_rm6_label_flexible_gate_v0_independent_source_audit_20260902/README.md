# STRATA-RM6 v0 independent source audit

This directory independently audits the externally pinned source-only package
`../strata_rm6_label_flexible_gate_v0` without opening Qwen, coarse, or matched
control payloads. The source package is treated as immutable.

Pinned inputs:

- source manifest SHA-256:
  `c8d56e045159e3af613f02c4d5d97c70e8f8b4383b3fbf282d384b08f74b7300`
- source root SHA-256:
  `d17718615dedebca08ead66c0555e9d649768a353f3a55d169a9bf400f11bd32`
- frozen STRATA independent auditor SHA-256:
  `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e`

The audit covers:

- six level-major planes and exact LSB-first indices 0 through 63;
- RM row orientation against the frozen independent STRATA decoder;
- canonical arithmetic coding, literal packet replay, CRC and padding attacks;
- bank dimensions versus emitted arithmetic bits;
- exact 2.5-bpw boundary rejection and sub-2.15 target ineligibility;
- zero-frozen versus current-random affine cosets;
- an independently implemented small-N exhaustive oracle;
- global exact-RM versus RM-ordered truncated-polar terminology;
- source manifest closure;
- CuPy execution-receipt scope;
- omitted outer expert container, inverse transforms and routed-read evidence.

It intentionally does not rely on the source author's `RED_TEAM.md` or any
same-agent audit directory.

Example source-only execution:

```bash
python -I -B audit_strata_rm6_v0.py \
  --package ../strata_rm6_label_flexible_gate_v0 \
  --auditor ../../strata_v2_klt_mixed_independent_auditor_v1.py \
  --cupy-receipt /tmp/strata_rm6_cupy_receipt.json \
  --output AUDIT_RESULT.json
```

The definitive disposition is in `ASSESSMENT.md` and `AUDIT_RESULT.json`.
