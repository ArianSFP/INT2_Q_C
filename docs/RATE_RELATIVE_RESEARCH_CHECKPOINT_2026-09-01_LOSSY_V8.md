# Rate-relative research checkpoint: lossy-tail v8

Date: 2026-09-01
Checkpoint status: **authenticated early kill; no codec promotion**

## Objective held fixed

The final claim remains a literal post-training compressed object with

```text
2.15 <= R <= 2.5 physical bits/weight
F = MSE * 2^(2R) <= 0.8
s = -0.5*log2(F) >= 0.16096404744368115 bpw
maximum cold/page bytes read for one routed expert < 2x
```

Every support, symbol, centroid, header, table, alignment byte, checksum and
zero-filled capacity bit is charged.  The read gate is computed from the page
union needed by a cold routed expert; it is not a warm-cache estimate.

The independently decoded expert-affine baseline remains `R=2.5`, relative
MSE `0.030902167403153148`, `F=0.9888693569009007`, and
`s=0.008074080480766676`.  It therefore misses the final target despite its
already-passing `1.1694444444444445x` worst cold-page read.

## What v8 tested

Lossy-tail v8 asked whether stable large-magnitude weights contain enough
non-Gaussian information to pay for an expert-local tail layer in front of the
finite residual codec.  It used six authenticated auxiliary layer-15 Qwen
Up/Down expert pairs and four moment-matched Gaussian controls.  The frozen
grid covered:

- coordinate, contiguous block-16 and contiguous block-64 supports;
- three physical rates in the permitted interval;
- raw and transformed score domains;
- fixed and adaptive support/symbol profiles;
- literal finite FP16-centroid rows; and
- a deliberately impossible zero-tail-error envelope used only as an early
  kill upper screen.

The CuPy production run searched 732 Qwen profiles per physical rate and the
identical grid for every control.  It emitted 225 retained score/read ledgers
containing 2,700 embedded profile rows.  No finite residual container was
encoded after the favorable oracle fell far below the preregistered gate.

## Independently reproduced result

The strongest impossible row is the raw/adaptive zero-tail-error envelope at
2.5 bpw:

| Quantity | Audited value |
|---|---:|
| Raw Qwen `F` before matched calibration | `0.2153922717726082` |
| Raw Qwen `s` | `1.1074808038288768` |
| Matched-control mean `s` | `1.0960945078646624` |
| Calibrated `F` | `0.9843391684954526` |
| Calibrated excess `s` | `0.011386295964214366` |
| Fraction of required `s` | `7.0738%` |

Nearly all of the spectacular uncalibrated score is therefore the ordinary
Gaussian advantage of revealing tail values for free, not Qwen-specific
compressible structure.

The best finite row is the 2.15-bpw raw/adaptive FP16-centroid cell.  Its Qwen
`F` is `1.002225093570817`, Qwen `s` is
`-0.0016032826638503746`, and its matched excess is only
`5.1039e-11`.  Its final joint score is negative.  The independently selected
decision is consequently **`EARLY_KILL_FAR_SHORT`** and no finite residual
codec is warranted for this bounded scalar-tail family.

## Read-bandwidth result

All retained objects are expert-local.  Across every one of the 225 ledgers,
not merely the winning rows, the independent audit reproduced:

| Read measure | Worst value |
|---|---:|
| Logical compressed-byte amplification | `1.015752828100462x` |
| Cold 4-KiB page-union amplification | `1.022286902319691x` |

This decisively passes the `<2x` MoE requirement and shows that locality is no
longer the blocker for this architecture.  The missing quantity is
source-specific information gain.

## Evidence chain

The scientific producer is
[`research/lossy_tail_peeling_oracle_v8/`](../research/lossy_tail_peeling_oracle_v8/).
Its source, runtime and one-use launch boundaries were checked before the Qwen
run.  The exact production result and authorization are retained in the
[independent result audit](../research/lossy_tail_peeling_oracle_v8_result_audit_20260901/).
The compact machine-readable checkpoint is
[`results/qwen/rate_relative_research_checkpoint_lossy_v8/checkpoint_manifest.json`](../results/qwen/rate_relative_research_checkpoint_lossy_v8/checkpoint_manifest.json).

| Artifact | SHA-256 |
|---|---|
| Frozen producer launch manifest | `6c5f5cd05973dbc0bf16cd9ea39951e690b15e15e13e969d2a33823117c2aa94` |
| Independent source-audit manifest | `045eac28701decf60837be335cfdf316b3ab1650125bc2f3744f097c0e75bb87` |
| Runtime calibration receipt | `45862549f34530964c4f8f7a4134228ccf036a8de3534f23e63e07acde7985b3` |
| Independent runtime-audit manifest | `7d0fbd622fd061641a3c0b96f00d1c6d4bf9c4b5f785d664fd6c8268df2a4134` |
| One-shot authorization | `16bf378c1c6baa23eaff7054ca4c1b82fa06ec45bd89e152956d6a51c752d1ef` |
| Exact 3,800,771-byte result | `2f3ebe509fa3c78c2caf6084510bb14e9e2a2fef9cabbdb4b99c9b396a4bfdf9` |
| Result-audit manifest | `fdfd809308957b38289960e20e8277393080b50cdc1c254c348199a4f832a4f9` |
| Result-audit receipt | `46df33b485ed65f62c5444daacd5653f4e3c5de117d5847455678136feb9428a` |
| Result-audit verifier | `1024563d3dbde7b2236d58b152a67a7b7c09e9b3145f8a3254196a4f05fd27c9` |

The standard-library result verifier passes **207,877** checks, including all
225 score/read rows, all 2,700 profiles, 48 control-moment cells, 18 calibrated
rows, twelve source receipts and its own seals.  A second copy was placed in a
disjoint RunPod scratch tree beside hash-matched frozen dependencies and
replayed with `/usr/bin/python3 -B -I`; it returned
`PASS_V8_INDEPENDENT_RESULT_VERIFICATION` with the same hashes and numbers.

To replay from the audit directory:

```bash
python3 -B -I verify_result_audit.py
```

## Claim boundary and next frontier

This is a strong scoped negative result, not a universal tail converse and not
a compression success.  The independent result audit authenticates the exact
supplied result and arithmetic; it does not regenerate payload moments from
the original Qwen files, serialize every discarded search trial, or provide a
kernel file-open trace.

The next work therefore does not enlarge the scalar-tail grid.  It targets
decoder-generated dependencies that can add information without HBM bytes:
exact initialization/RNG reconstruction and a direct joint cross-role
SwiGLU path predictor.  Both must pass separate source and runtime audits,
matched controls, non-additive rebuilt-residual scoring, exact side-rate
charging and the same `<2x` read gate before any promotion.
