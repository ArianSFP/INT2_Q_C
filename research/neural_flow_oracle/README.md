# Continuous-flow and shared-expert early-kill oracles

This bundle records three bounded negative experiments against the Qwen3-30B-A3B
20%-below-Gaussian target.  They are source-only PTQ screens, not codec MSE
results and not universal rate-distortion converses.

The common identity is

```text
F = D * 2^(2R)
s = -0.5 * log2(F)
F multiplier from an effective rate advantage s = 2^(-2s)
required s for F <= 0.8 = -0.5 * log2(0.8)
                          = 0.160964047443681 bpw
```

Every JSON result embeds the SHA-256 of its executing script and every source
it was allowed to inspect.  The pinned 18-matrix panel is bound through its
source lock and individual matrix hashes.  Both flow gates failed before the
pinned files were opened.

## 1. Cross-fitted continuous long-context mixture/flow

`continuous_long_context_flow_oracle.py` uses 57 auxiliary Up/Down pairs after
removing every pair whose layer **or** expert ID occurs in the pinned panel.
For each of 12 folds, training again removes every pair sharing any test layer
or any test expert.  The scored sample contains 58,368 weights.

The representation receives an exact two-role KLT and exact role-by-stratum
means/scales for free.  The strongest oracle also receives exact future
weights, contexts extending 1,024 columns and 256 rows, and the paired KLT role.
Model and side bytes are uncharged.  This deliberately favors the candidate.

| Variant | Free-side gain (bpw) | Two-SE upper bound (bpw) | `F` multiplier |
|---|---:|---:|---:|
| Marginal Gaussian mixture, 8 components | **0.0001452778** | **0.0004898386** | **0.9997986225** |
| Causal self-context affine flow | -0.0020404865 | -0.0010556924 | 1.0028327195 |
| Bidirectional self-context affine flow | -0.0020070982 | -0.0008534868 | 1.0027863035 |
| Bidirectional cross-role affine flow | -0.0030223402 | -0.0016705478 | 1.0041986428 |

The best point estimate supplies only 0.0903% of the required `s`; even its
two-SE upper bound is 328.6 times too small.  Decision:
`HARD_KILL_CONTINUOUS_LONG_CONTEXT_FLOW`.

## 2. Actual learned nonlinear affine flow

`nonlinear_mlp_affine_flow_gate.py` is a separate learned nonlinear check, not
the fixed-feature ridge screen above.  A `32 -> tanh(32) -> tanh(16) -> 2`
network predicts conditional mean and bounded log variance from long-range
continuous neighbours.  It trains only on layer-15 auxiliary experts.  Twelve
experts train the network; experts `{24, 56, 88, 120}` are untouched
validation.  An iid Gaussian control has exactly the same array dimensions,
moment matching, initialization, optimizer, updates, and stop rule.

The predeclared bounded rule stops after 80 updates when the best auxiliary
validation gain remains below 0.05 bpw:

| Quantity | Result |
|---|---:|
| Qwen auxiliary validation gain | -0.0351209692 bpw |
| Matched-Gaussian control gain | -0.0323597785 bpw |
| Control-adjusted gain | -0.0027611907 bpw |
| Pinned matrices opened | No |

The frozen FP16 decoder contains 1,682 values and is exactly 3,428 bytes.  With
96 local stratum-moment bytes per expert, its conservative charge over only the
six-expert target panel is 0.0011314110 bpw.  Amortized over 128 experts, a cold
expert read would be 1.0023064070x at 2.5 bpw; a resident decoder is effectively
1x.  These favorable bandwidth figures do not rescue a negative information
gain.  Decision: `HARD_KILL_NONLINEAR_FLOW_BEFORE_PINNED`.

## 3. Same-layer shared expert templates and procedural bases

`shared_expert_basis_oracle.py` audits all 16 sampled layer-15 experts
`{0, 8, ..., 120}` for both Up and transposed Down.  Every learned basis excludes
the evaluated expert.  FP64 Gram projection treats the learned basis as exact;
quantization error is ignored before the side charge, making this an optimistic
oracle.

| Candidate | Residual energy `q` | Oracle `s` (bpw) | Optimistically charged `s` (bpw) | Cold read amp |
|---|---:|---:|---:|---:|
| Leave-one-expert-out mean template | 1.0667931973 | -0.0466402651 | -0.0664970360 | 1.984368945x |
| Best learned rank-1 basis | 0.9999993076 | 0.0000004994 | -0.0198664439 | 1.984364971x |
| Full rank-15 span of all other experts | 0.9999871876 | 0.0000092423 | -0.2934376164 | 14.319398267x |
| Analytic zero-read DCT basis, rank 64 | 0.9999572558 | 0.0000308341 | -0.0009457284 | 1.000000000x |

For learned bases, the optimistic charge assumes each exact basis matrix costs
only 2.5 bpw amortized over 128 experts while ignoring its quantization error.
The JSON also reports the stricter exact-BF16 charge.  Rank 1 is the only learned
full-matrix basis below 2x cold reads, and its oracle signal is only 0.00031% of
the required `s`.  The analytic basis needs no basis reads but still loses after
its FP16 coefficients and 64-byte local frame.  Decision:
`HARD_KILL_SHARED_EXPERT_BASIS`.

## Reproduction

The broad continuous screen was run locally because the complete 68-pair
auxiliary cache was present there:

```powershell
python continuous_long_context_flow_oracle.py `
  --dev-dir C:\INT2__compression\qwen_weight_cache\tensors `
  --target-lock C:\INT2__compression\INT2_Q_C\blind_protocol_v2\unblinded\source_hashes.lock.json `
  --output continuous_long_context_flow_oracle.json `
  --samples-per-role 512
```

The nonlinear and shared-expert screens were run CPU-only on the supplied
RunPod, where the 16-expert layer-15 cache resides:

```bash
/workspace/int2-cupy-venv/bin/python nonlinear_mlp_affine_flow_gate.py \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --target-dir /workspace/INT2__compression/blind_protocol_v2/unblinded/sources \
  --target-lock /workspace/INT2__compression/blind_protocol_v2/unblinded/source_hashes.lock.json \
  --output-dir nonlinear_mlp_affine_flow_gate \
  --samples-per-matrix 4096 --max-steps 160 --batch-size 2048

/workspace/int2-cupy-venv/bin/python shared_expert_basis_oracle.py \
  --source-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --output shared_expert_basis_oracle.json
```

## Exact claim boundary

These results reject the tested families: cross-fitted conditional affine
flows with the listed long/bidirectional/cross-role features, the bounded
two-layer MLP conditional flow, shared full-matrix expert PCA/template bases,
and the listed analytic DCT modes.  They do **not** prove that arbitrary neural
flows, nonlinear manifolds, functional-equivalence transforms, or a different
joint quantizer cannot reach `F <= 0.8`.
