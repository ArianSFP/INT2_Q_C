# Qwen categorical bitplane/context probe — early-kill report

## Verdict

Stop this branch. A materially different global-RMS, symmetric Lloyd-4 label
construction found no useful higher-order categorical redundancy on disjoint
Qwen MoE layers/experts.

The best legal, table-charged cross-fit result was the exact `(left, above)`
context:

- raw saving: **0.0001303590604 bpw**
- saving after the dense 12-bit probability-table charge:
  **0.00008959600138 bpw**
- ideal entropy-rate factor implied by that saving:
  **`F = 2^(-2s) = 0.999875801281804`**
- required saving for `F <= 0.8`: **0.160964047443681 bpw**
- charged result as a fraction of the gate: **0.0556621%**

Even the most favorable deliberately leaky/optimistic measurement across all
new tests was only **0.003708256572 bpw**. That would imply
`F = 0.994872455790230`, reaches only **2.30378%** of the required entropy
advantage, and remains **0.157255790872 bpw** short of the gate before a real
entropy coder or additional implementation costs.

This agrees with, but does not duplicate, the earlier full 136-matrix audits:
the prior tile-normalized 32-symbol causal MLP produced -0.000221 bpw and the
best prior exact nonlocal table produced 0.004725 bpw raw / 0.003500 bpw after
its table charge. Together, the full prior evidence and this different frozen
label/context construction justify killing the family without spending a full
run on marginal variants.

## Frozen protocol

- CPU-only NumPy; no CuPy/CUDA import and no GPU work.
- 44 auxiliary Up/Down tensors, 22 layer-experts, and 69,206,016 weights.
- Eight layers selected by a fixed evenly-spaced depth rule before scoring:
  `0, 6, 13, 18, 26, 31, 39, 47`.
- Whole-layer two-fold split: every expert and role in a layer stays in one
  fold. Each Lloyd codebook and conditional table sees training layers only.
- Up uses native `(768, 2048)` orientation; Down uses the globally fixed
  transpose so its intermediate-neuron axis is aligned. There is no
  data-selected orientation bit.
- Each matrix is normalized by its own RMS. Symmetric two-magnitude Lloyd
  centroids and thresholds are frozen from training tensors, then applied to
  held-out tensors. Held-out source-relative MSE is **0.128489970453254**.
- Cross-fit conditional probabilities use fixed strength-32 Dirichlet backoff.
  Dense probability tables are charged at 12 bits per nonredundant
  probability plus framing. FP16 matrix scales, matrix framing, and stream
  termination are included in the absolute coarse-code rate ledger.
- The float cross-entropy omits arithmetic-coder redundancy and is favorable
  to the candidate. The held-out plug-in values are explicitly leaky,
  downward-biased opportunity screens; they are used only for an early kill.

## New contexts tested

The ordinary categorical contexts and results were:

| Context | Raw cross-fit gain (bpw) | Charged gain (bpw) | Optimistic plug-in gain (bpw) | Ideal F from charged gain |
|---|---:|---:|---:|---:|
| Left | 0.0000610592 | 0.0000453304 | 0.0001246052 | 0.9999371607 |
| Above | 0.0000711015 | 0.0000553598 | 0.0000944127 | 0.9999232579 |
| Left + above | 0.0001303591 | **0.0000895960** | 0.0002196067 | **0.9998758013** |
| Left + above + upper-left | 0.0001261139 | -0.0000147039 | 0.0002262753 | 1.0000203841 |
| Exact causal five: L, L2, U, UL, UR | 0.0000739377 | -0.0020700705 | 0.0004995973 | 1.0028738487 |
| Fixed 4x4 semantic position | -0.0000129451 | -0.0000536352 | 0.0000121222 | 1.0000743570 |
| Left + above + 4x4 position | 0.0000855995 | -0.0004554375 | 0.0002632853 | 1.0006315698 |

Two additional nonlinear constructions were screened generously:

- Separate magnitude and sign planes: magnitude conditioned on five causal
  magnitudes, then sign conditioned on the current magnitude and five causal
  signs. Its best optimistic opportunity was **0.0003799532982 bpw**.
- Expert-local cross-role context: Down-transpose conditioned jointly on the
  aligned Up label and Down's own L/U/UL labels. Its best deliberately
  favorable opportunity was **0.003708256572 bpw** over the two-role payload.

These are materially different from the earlier tile-normalized neural model:
the labeler uses a matrix RMS and frozen Lloyd reconstruction points, the
neighborhood is an exact 2-D categorical state, and sign/magnitude bitplanes
are factorized explicitly.

## Read behavior

All legal causal models read only the current expert's stream plus a shared
probability table. Warm-cache expert read amplification is exactly **1.0x**.
For the largest ordinary table (the exact five-neighbor Lloyd-4 context), the
measured conservative cold-table amplification is **1.0124266671x** for the
two-matrix auxiliary panel and **1.0082844447x** for a full three-matrix
expert. Thus read locality is excellent; the branch fails solely on MSE/rate
opportunity.

## Reproduction artifacts

- `agent_bitplane_context_probe_pilot_frozen.py` is the exact script executed
  for the frozen pilot. SHA-256:
  `2071463c85147493c2026f8aa20fbad0e265773f54fd58f1837be1c0fbdb4d94`
- `agent_bitplane_context_pilot.json` contains source hashes, fold assignment,
  frozen codebooks, per-direction metrics, charges, and read accounting.
  SHA-256:
  `d1da0a02c118d848377e8e36a373da1d8cc6a68f3d4b2b91ed07f9037c46d4bb`
- `agent_bitplane_context_probe.py` contains a post-run audit improvement that
  makes the sign/magnitude and cross-role summaries legal-direction-only. It
  was not needed to make the negative decision and was intentionally not run
  after the early-kill gate fired.

Frozen command:

```bash
/workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/agent_bitplane_context_probe.py \
  --tensor-dir /workspace/INT2__compression/qwen_aux_context_tensors \
  --output /workspace/INT2__compression/agent_bitplane_context_pilot.json \
  --alphabets 4 --sample-per-tensor 16384 --lloyd-iterations 10 --pilot
```

The final decision is not a theorem about arbitrary learned weight models. It
is a strong negative result for local categorical contexts up to five causal
neighbors, semantic position bins, explicit sign/magnitude bitplanes, and the
tested richer expert-local Up-to-Down context.
