# PMG1 binary64 / explicit-FMA v2 independent source audit

Verdict: **BLOCK payload authorization; authenticate the v2 stage-0 survivor.**

The standard-library-only audit independently reconstructed the exact ABI1
bundle and full plan, both CUDA translation units, the v2 wrapper/template,
all explicit binary64 contraction sites, every packed journal record, the
canonical TopK and synthetic global merge, projection arithmetic, runtime file
and header hashes, and the three parity receipt families. It executed no
framework, accelerator, network, or model operation.

The measured stage-0 decisions are valid within their narrow scope:

- binary64-v1: `696.8922319519334 > 650`, so it is killed;
- explicit-FMA-v2: `520.8358833260136 < 650`, so it survives stage 0;
- v2 materially changes the arithmetic: all 8192 seeds remain in the TopK set,
  their order changes, and 119 common-seed capture encodings change. The best
  seed/capture and the scaled-BF16/Torch-state receipts remain unchanged.

The decisive scientific blocker is stronger than the draft's generic missing
attestation. Experts `[24,56,88,120]` are not untouched: two earlier frozen
result receipts, hashes `6ef38ff1...` and `e450c107...`, already serialize
target-derived fit/score moments for all eight matrices. The v2 validation
claim therefore cannot be authorized or repaired retrospectively.

A distinct v3 with source-independently precommitted, genuinely unopened
experts is directionally sound only if the identities are outside every prior
cache/result/log and remain inaccessible until one descriptor, all selection
state, controls, thresholds, and a no-retry sentinel are durably sealed. It
still needs retention, complete tail timing, complete compiler/runtime bytes,
and crash-safe 256-shard/two-tree merge evidence.

Run on the bound RunPod source tree with:

```text
/usr/bin/python3 -B -I research/fuseed_pmg1_binary64_fma_calibration_v2_independent_source_audit/audit.py
```

The program recomputes and compares the sealed receipt when
`audit_receipt.json` is present. See that receipt for the exact 148185-check
ledger and limitations.
