# Conditional hyperprior early-kill probe

This directory contains a CPU-only, expert-local information probe for the
pinned six-triplet Qwen3-30B-A3B panel. It was deliberately kept outside
`INT2_Q_C`; the compression repository was not changed.

## Outcome

The conditional scale-hyperprior branch is rejected before pinned-source or
GPU evaluation.

| Quantity | Result |
|---|---:|
| Required structural gain | 0.1609640474 bpw |
| Best cross-fitted gross signal gain | 0.0062288368 bpw |
| Expert-local physical side cost | 0.0008262054 bpw |
| Shared-model cost, charged over the target panel | 0.0004679362 bpw |
| Best physically charged net gain | **0.0049346951 bpw** |
| Fraction of requirement | 3.0657% |
| Implied multiplier on `F = D 2^(2R)` | 0.9931824 |
| LEO+LLO folds with negative local net gain | 40 of 57 |

The selected development candidate uses a six-bit residual log-RMS code per
2,048-weight natural group with a 0.25-log2 step. The frozen canonical Huffman
model is 1,656 bytes. The scheme would preserve 1.0x steady-state expert
payload locality if its side stream were colocated with each expert and the
small shared model were cached, but the information benefit is immaterial.

## Leakage controls

The target consists of layers `{5, 12, 18, 28, 36, 45}` and expert IDs
`{7, 18, 20, 41, 76, 83}`. Training globally excludes every auxiliary pair
whose layer is in the target-layer union **or** whose expert ID is in the
target-expert union. This leaves 57 auxiliary Up/Down pairs.

For each development holdout `(layer, expert)`, its conditional-code table is
fitted after subtracting every remaining pair with the same layer **or** the
same expert ID. Thus reported model-selection scores are
leave-expert-and-layer-out rather than random-tile splits.

The 105 candidates are the Cartesian product of:

- tile sizes `32, 64, 128, 256, 512, 1024, 2048`;
- code widths `2, 3, 4, 5, 6`;
- residual log2-scale steps `0.125, 0.25, 0.5`.

Selection reads only development sources. It writes and hashes the model and
freeze before the `evaluate` subcommand is allowed to open a pinned source.
The required gate failed by 0.1560293523 bpw, so the pinned evaluation was not
run.

## Metric and physical ledger

For each expert, Up and transposed Down are transformed by an exact two-channel
KLT. Natural 2,048-value rows are energy-ranked into eight equipopulous STRATA
bins. The baseline is deliberately strong: it receives an exact target-derived
FP16 Gaussian scale for every role-by-stratum cell for free in the NLL
comparison. The candidate sends a quantized residual tile scale and uses the
resulting conditional Gaussian density.

The measured signal term is held-out baseline Gaussian cross-entropy minus
conditional Gaussian cross-entropy. The following bytes are then subtracted:

- canonical Huffman payload and byte padding;
- independently framed expert-local headers;
- all FP16 context scales;
- integrity and context hashes;
- the complete shared canonical Huffman model, charged over only the pinned
  28,311,552 weights even though it may be cached during inference.

This is an information/RD screen, not a reconstructed-weight MSE result. A
passing result would still require integration into an operational quantizer
and independent source-domain decoding.

## Predeclared stop rule

- Kill if physically charged pinned net gain is below
  `-0.5 log2(0.8) = 0.16096404744368115 bpw`.
- Promote only if net gain is at least `0.18096404744368115 bpw` (a 0.02-bpw
  implementation margin) and every expert is locally positive.
- A development score only 3.07% of the hard gate triggers the earlier,
  cheaper stop before pinned-source access.

## Artifacts

- `conditional_scale_hyperprior_probe.py`: train/evaluate implementation;
- `frozen/conditional_hyperprior_freeze.json`: all source hashes, folds,
  candidate aggregates, and gate definitions;
- `frozen/conditional_hyperprior_model.bin`: canonical cached decoder model.

Identities:

- freeze SHA-256:
  `f78fe27314eebfde4ffe412c6d51a0f4081cbfcd06e40c79b95de8fbbec4bb80`;
- model SHA-256:
  `59cba572eebe0133ac1d378650f20f16ab4d250c99abfee44ebbad295307c3e7`.

The completed development command was:

```powershell
python conditional_hyperprior_probe/conditional_scale_hyperprior_probe.py train `
  --dev-dir qwen_weight_cache/tensors `
  --target-lock INT2_Q_C/blind_protocol_v2/unblinded/source_hashes.lock.json `
  --output-dir conditional_hyperprior_probe/frozen
```

The hash-gated `evaluate` command remains implemented for a future materially
different hyperprior that first survives the development threshold.
