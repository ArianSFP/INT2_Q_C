# FUSEED-PMG1-v3 fresh-validation precommit

This source-only precommit repairs a fatal chronology problem in the PMG1-v2
draft: its four proposed validation experts were members of an auxiliary cache
already opened by several earlier studies. They cannot honestly be called an
untouched one-shot panel.

Before reading any replacement tensor payload, this package fixes a public,
source-independent choice. Among layer-15 expert IDs not divisible by eight,
rank the SHA-256 hashes of
`FUSEED-PMG1-v3|fresh-validation|layer=15|expert=E` and take the four smallest.
The result is experts `67, 95, 69, 34` in rank order. Gate, Up, and Down are all
included so a surviving descriptor cannot defer its Gate-half test to another
adaptively chosen panel.

`validation_precommit.json` binds the pinned Qwen revision, the already-public
safetensors index and header metadata, all twelve tensor names, shapes, shards,
relative offsets, and exact inclusive HTTP ranges. Only this metadata was
inspected. None of the twelve tensor ranges was fetched or materialized by the
precommit turn.

This file does not authorize a fetch. A distinct v3 design must bind the
explicit-FMA arithmetic, coordinate plan, retention suite, full-pipeline
calibration, one-descriptor commit, and no-retry firewall. An independent
source audit must pass before selection; the twelve ranges stay inaccessible
until that selection is durably sealed.

Verify locally from the repository root:

```powershell
pwsh -NoLogo -NoProfile -File research/fuseed_pmg1_v3_fresh_validation_precommit/verify_precommit.ps1
```

