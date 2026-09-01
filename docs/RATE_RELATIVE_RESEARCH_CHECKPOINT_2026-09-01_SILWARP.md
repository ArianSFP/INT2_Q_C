# Rate-relative Qwen research checkpoint — SILWARP and ancestry gates

## Status

This checkpoint preserves the completed work after the first 2026-09-01
rate-relative research checkpoint. It is a negative-result checkpoint, not a
claim that the final objective has been reached.

The acceptance rule is unchanged:

```text
2.15 <= R_actual <= 2.5 bpw
F = MSE * 2^(2 R_actual) <= 0.8
s = -0.5 log2(F) >= 0.16096404744368115 bpw
maximum cold compressed read per routed expert < 2x
```

The finite STRATA expert-affine checkpoint remains the operational baseline:
`R=2.5`, MSE `0.030902167403153148`, `F=0.9888693569009007`, and
worst page-exact cold read `1.1694444444444445x`. The required same-rate MSE
is `0.025`.

## Completed results

| Branch | Strongest authenticated result | Decision |
|---|---:|---|
| Tier-B initializer ancestry | corrected validation capture `-0.0013008376`; corrected `+3 SE=0.0016152600` versus `0.1456888484` required | Hard kill for 37,748,736 logical MCore keys |
| SILWARP-v2 recurrent hyperdecoder | exact identity selected by both fixed seeds at updates 256 and 512; `s_match_worst=0`, cluster SE `0` | Hard kill for the frozen cell |
| Scalar rotated-dither finite bridge | system rate `2.4141676956`; identity MSE `0.0507005875`; 44.46% correction required | Kill as a finite bridge |
| Public dense-to-MoE ancestry | raw capture `0.0051296784`; corrected capture `0.0001263806` versus `0.1456620755` required | Scoped kill for Qwen3-1.7B layers 9/15 |
| Permutation-aligned pair | raw capture `0.0029652832`; corrected `0.0003750979` | Heuristic pause of this one unweighted-cosine joint-scalar cell only |

### Tier-B grouped-ancestry predecessor

The sealed Tier-B search evaluated 37,748,736 logical expert-major
Megatron-Core/PyTorch-CUDA keys with 32 matched null searches and four
untouched whole-expert folds. Its source winner did not beat the nulls, two
whole-expert folds were negative, and both role folds were negative. The
result SHA-256 is
`e450c10767b54c190f901df8460c6ac57fe86cfaca7719c3db48475d4196fb92`.

This rejects only the sealed expert-major family. It does not identify Qwen's
training framework, seed, stream offset, or production initializer.

### SILWARP-v2 ideal-channel gate

SILWARP retains a full-rate Gaussian test-channel reconstruction and applies a
shared 235,779-parameter, six-step nonlinear tile warp. Its source-derived
means and RMS values, model bytes, headers, and cold reads are explicitly
charged. At the 128-expert ledger it has physical rate
`2.15643318494161 bpw` and cold read `1.37104489269124x`.

The first frozen execution stopped before training because mathematically
equal signed FP16 zeros had different bit patterns. V2 made only the numerical
no-op repair of canonicalizing serialized zero means. The repaired CuPy run on
the RTX 5090 then reached the preregistered update-512 gate. Both fixed seeds
selected the exact identity bypass at updates 256 and 512, with zero matched
gain and zero delete-cluster jackknife SE. Confirmation and the pinned panel
remained closed.

The result SHA-256 is
`c21567d337f6349f31c8c8f0eaa2a544f54a772afceb3cf607bcc10119a84592`.
The independent authenticated-hard-kill receipt SHA-256 is
`0489c6d17dbe6be7c565319005c3bf3cd3c5acd02c0afc940ab719c8c8695b20`.

Evidence:

- [`research/implicit_hyperdecoder_gate/`](../research/implicit_hyperdecoder_gate/)
- [`research/implicit_hyperdecoder_gate_v1_failure/`](../research/implicit_hyperdecoder_gate_v1_failure/)
- [`research/implicit_hyperdecoder_gate_v2/`](../research/implicit_hyperdecoder_gate_v2/)
- [`research/implicit_hyperdecoder_gate_v2_result/`](../research/implicit_hyperdecoder_gate_v2_result/)
- [`research/silwarp_v2_result_audit/`](../research/silwarp_v2_result_audit/)

### Finite-channel bridge

A source-free production-shape serializer tested whether scalar
rotated-dither quantization could turn an ideal nonlinear survivor into a
finite stream. The actual `N=2^19` arithmetic payload used
`2.4075832367 bpw`; the FP16 decoder and table raise the system rate to
`2.4141676956 bpw`, with cold read `1.3344648962x`. Its measured identity MSE
is `0.05070058749`, while the same-rate target is `0.02815893369`. A decoder
would therefore need a 44.46% correction, or `s=0.4242048804 bpw`.

This kills scalar rotated dither as the bridge. It does not kill direct finite
POLARIS reconstruction: the retained implementation plan uses nine private
`N=2^19` streams per expert and list analysis-by-synthesis over legal polar
codewords, with no transmitted list index. No positive finite result is
claimed.

The synthetic result SHA-256 is
`442e34177df01ad898d11f982c99b555f65e9a70cf3fbda8fda6424c57bbaeef`.
Evidence and exact ledgers are in
[`research/finite_channel_bridge/`](../research/finite_channel_bridge/).

### Structural ancestry screens

The public dense-reference screen gives exact Qwen3-1.7B-Base layer-9 and
layer-15 Up/Down matrices to a fixed non-pinned Qwen3 MoE expert for free. It
selects the better layer, solves a rectangular neuron assignment, and fits
per-role affine coefficients without charging any reference or side bytes.
The raw opportunity is already 28.40 times too small; after four matched
scramble controls it is 1,152.57 times too small. The scoped independent audit
passes. Result SHA-256:
`36e7eb51f3eef51f88e6b08c562905c0e2797949b18327c35e2033363cd5db71`;
audit SHA-256:
`b3015194876a7504e90334e2eade9ab52db2bfd891e680e8b625a867fbda9fb5`.

The separate permutation-aligned pair is retained as heuristic evidence, not
as an upper bound. Independent review found that its Hungarian objective
maximizes unweighted cosine squared, whereas explained raw SSE also contains
the target-neuron energy. It also omits Gate and separate role coefficients.
The recorded numerical cell is reproducible, but its original family-kill
wording is superseded by the audit. Result SHA-256:
`ba22a5ac76a6cc697f63899787ab85396a5b00dc2d764299473ecb59e3a52a52`;
audit SHA-256:
`d375777d76f4b89e92164ea2e36fc0bcb47fe66dacd5e68d68599fcdf0526ed2`.

Evidence:

- [`research/dense_upcycle_reference/`](../research/dense_upcycle_reference/)
- [`research/permutation_aligned_expert_template/`](../research/permutation_aligned_expert_template/)
- [`research/structural_reference_audit/`](../research/structural_reference_audit/)

## Verification performed

The checkpoint was rechecked locally with:

```text
SILWARP-v1 source-free tests: 32/32 pass
SILWARP-v2 source-free tests: 35/35 pass
Finite-bridge source-free tests: 5/5 pass
SILWARP result verifier: PASS_SILWARP_V2_NEGATIVE_RESULT
SILWARP independent receipt: seal valid
Structural package audit: all 41 checks pass
Structural receipt seals: both valid
```

Raw run logs are marked `-text` in `.gitattributes`, and checksum sidecars are
fixed to LF, so content-addressed evidence is not rewritten by checkout line
ending conversion.

## Active work deliberately excluded

The Tier-C projection-major `TEGroupedMLP` initializer gate and the broader
procedural seed/offset expansion remain unsealed work in progress and are not
part of this checkpoint. Tier-C tests the source-backed FC1-all-experts then
FC2-all-experts call order missing from Tier B. The broader expansion is also
required to cover Hugging Face global streams, because Tier A searched only
seven common seeds.

Neither directory may access Qwen payloads or CUDA until its source-only lock,
parity gates, calibration protocol, and independent audit are complete. A
positive initializer survivor would still need a separately frozen Gate-role
confirmation and a finite residual codec before it could affect the final
claim.
