# POLARIS-SC-v2 on real Qwen weights

Date: 2026-08-31

## Verdict

The unpreconditioned frozen **POLARIS-SC-v2 fails** on the real-Qwen coverage
panel even though its rate fits. Its serialized, energy-weighted relative MSE
is `0.06319873774126093`, or `24.4908%` above the 2.15-bpw matched-Gaussian
limit.

The deployment-complete **POLARIS-SC-v2 + deterministic randomized Hadamard
preconditioner passes**. Its independently decoded relative MSE is
`0.05289448474927123`, or `4.19320%` above the Gaussian limit, under an exact
emitted rate of `2.14971923828125 bpw`. The declared target was no more than
`5%` above the Gaussian limit.

This is a deterministic 32-block coverage-panel result, not a full-checkpoint
census or a statistical confidence claim.

## Frozen test

- Checkpoint: `Qwen/Qwen3-30B-A3B`
- Revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Manifest SHA-256:
  `3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55`
- Values tested: `32 * 262144 = 8,388,608` exact BF16 weights
- Coverage: six blocks from each expert projection (gate/up/down), two blocks
  from each attention role (q/k/v/o), and two each from embeddings, LM head,
  and routers
- Seeds: deterministically re-derived from
  `SHA256(revision:tensor:canonical_block_index)` and checked for collisions
- Hardware/runtime: NVIDIA RTX 5090, CuPy 14.2.0, four concurrent GPU encoders
- Serialization: fixed POLARIS v2 overflow reservoir, u32 length and FP16 scale
  per block, independently unpacked and decoded

The expert blocks deliberately avoid expert 31, which had been used during
development. The panel is not fully blinded: embedding block 0 was used for a
pipeline/RHT round-trip smoke before the panel run. No quantizer parameter,
sample membership, SC seed, or retry was changed after observing outcomes.

## Primary results

| Metric | Exact v2 | v2 + deterministic RHT |
|---|---:|---:|
| Logical payload mean (bits/block) | 559,903.3750 | 562,697.3125 |
| Payload capacity (bits/block) | 563,464 | 563,464 |
| Pooled logical headroom | 113,940 bits | 24,534 bits |
| Emitted physical reservoir | 2.1497192383 bpw | 2.1497192383 bpw |
| Energy-weighted relative MSE | 0.0631987377413 | 0.0528944847493 |
| Excess over Gaussian limit | 24.4908% | **4.19320%** |
| Passes 5% Gaussian gate | No | **Yes** |
| Fresh independent decodes | 32/32 | 32/32 |
| Individual blocks under 5% ceiling | 27/32 | **32/32** |

Reference values:

- Gaussian MSE limit at 2.15 bpw: `0.050765774772264724`
- Five-percent ceiling: `0.053304063510877964`
- Maximum RHT block MSE: `0.0530559104446109`
- RHT reduction versus exact v2 on identical sources: `16.3045%`

A descriptive role-population expansion gives RHT MSE
`0.05286612430799479` (`4.13733%` above Gaussian) and logical payload
`562734.4402` bits/block. This expansion uses exact checkpoint block counts,
but it is not a probability-sample estimator and has no design-based
confidence interval.

## Bottleneck and breakthrough

The exact codec's failure is distribution mismatch, not insufficient average
rate. Raw Qwen kurtosis predicts its block error almost perfectly on this
panel (`r = 0.99521`). The main failures are:

| Block/role | Raw kurtosis | Exact MSE | RHT kurtosis | RHT MSE |
|---|---:|---:|---:|---:|
| L47 attention-o, block 31 | 114.292 | 0.19918145 | 2.99286 | 0.05284034 |
| Embedding, block 0 | 31.311 | 0.09489737 | 3.00918 | 0.05290561 |
| L0 router | 24.481 | 0.07827474 | 2.99597 | 0.05285112 |
| L47 router | 15.434 | 0.05803173 | 3.01719 | 0.05283185 |

For the worst attention block, only the top `0.1%` of weights contain
`20.33%` of source energy and the maximum magnitude is `42.20 RMS`. The
Gaussian-matched polar test channel is a poor direct model for that source.

The zero-side-bit preconditioner uses

`y = H_N diag(s) x / sqrt(N)`

where `s` is a deterministic Rademacher diagonal derived from the canonical
block identity. Decoding uses

`x_hat = diag(s) H_N y_hat / sqrt(N)`.

Across all 32 blocks, raw median kurtosis is `3.37547` and the maximum is
`114.292`. After the transform, median kurtosis is `3.00146` and the maximum
is only `3.01946`. The RHT therefore makes the source actually match the
Gaussian-designed codec while preserving energy and adding no per-block side
bits. Post-RHT error no longer correlates with kurtosis (`r = -0.1212`).

## Rate and integrity checks

- Exact file size: `18,033,152` bits / `2,254,144` bytes for both arms
- Required fixed size: `768 + 32 * (563464 + 48) = 18,033,152` bits
- Exact 2.15 gate checked with integer arithmetic:
  `physical_bits * 20 <= 43 * source_values`
- RHT reservoir SHA-256:
  `55d347c02ef1382ce209050d539f4e336dd7477125e4319e8b78d3067a436aac`
- Exact-v2 reservoir SHA-256:
  `9388790c3cdbab5b9b33b676ced196090d81ba0422eb6fecfdd014bd2d054cf5`
- Every source hash, canonical identity, derived seed, logical payload,
  padding bit, FP16 scale byte, causal frequency stream, reconstruction index,
  inverse RHT, and final FP64 reconstruction hash was checked fail-closed.

Pinned implementations:

- Frozen exact encoder: `95cfd32e...d9b8`
- RHT encoder: `062f74ca...78a0`
- Evaluation runner: `4229ffd0...3d59`
- Packer: `c5fda342...dde8`
- Unpacker: `cf7113c3...7e07`
- Independent decoder: `2e1e484b...c797`
- Decoder map: `a0e9895d...8ef`

## Claim boundary and next test

This result establishes that the RHT-completed codec meets the 5%-of-Gaussian
MSE goal on a broad real-Qwen matrix panel under an actually emitted
2.15-bpw reservoir. It does not establish full-checkpoint rate/MSE,
perplexity, downstream accuracy, or runtime efficiency. The tested values are
about `0.0275%` of the checkpoint's rank-2 weights.

The decisive next test is a probability sample or complete census of all
`116,470` rank-2 blocks, followed by full checkpoint reconstruction and model
evaluation. Because every tested RHT role is tightly clustered near the same
Gaussian operating point, scaling the census is now an engineering and
statistical-validation task rather than an unresolved codec-design failure.

## Artifacts

- `qwen_polaris_heldout32_manifest.json`
- `qwen_polaris_heldout32_results/exact_v2/summary.json`
- `qwen_polaris_heldout32_results/rht_v2_postrms/summary.json`
- `qwen_polaris_heldout32_paired_audit.json`
- `qwen_polaris_heldout32_distribution_audit.json`
- `agent_polaris_qwen_paired_audit.py`
- `agent_polaris_qwen_distribution_audit.py`
- `agent_polaris_qwen_panel_runner.py`
- `agent_polaris_qwen_rht_encoder.py`
- `agent_novel_qwen_reservoir_decode.py`

The independent paired-audit JSON SHA-256 is
`666fa77eb17fc9fda2c27f3081913b7a383d103acaf9b92749fcbbe921bcfedc`.
The CuPy distribution-audit JSON SHA-256 is
`f4724f0bd1118074b28ffed50d635a75640c5b926bf70777bfe7e1dc3e437fa0`.
