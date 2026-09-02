# TACTIC-DH384 finite v3 Qwen pilot result

Date: 2026-09-02

Status: **hard rejected by the continuous parent-span gate; no finite
composite emitted**.

The independently audited `307/128`-bpw coarse artifact was decoded against
the exact externally pinned BF16 Gate, Up, and transposed-Down source. The
frozen dyadic basis was then scored before any finite 384-bit engineering.

## Result

- Coarse relative MSE: `0.03697515006059532`.
- Coarse SSE: `89.64596664948954`.
- Exact capture required for `D <= 0.025` at literal 2.5 bpw:
  `0.3238702220537387` (32.3870222%).
- Rank-384 continuous projected energy: `8.392531479313217`.
- Rank-384 measured capture: `0.09361861769116196` (9.3618618%).
- Implemented rank-376 projected energy: `8.216354331023433`.
- Rank-376 measured capture: `0.0916533630916034` (9.1653363%).
- Isotropic rank-384 dimension fraction: `0.09375`.
- Decision:
  `HARD_REJECT_PARENT_RANK384_CONTINUOUS_SPAN_BELOW_NESTED_THRESHOLD`.

The measured parent capture is essentially the isotropic dimensional share,
so this fixed dyadic subspace did not expose material Qwen-specific residual
alignment. Because every finite v3 correction lies in the rank-376 subset of
that failed parent span, the encoder stopped without creating a composite.

## Physical and claim boundary

The source package and RunPod CuPy mechanics were previously verified, and
the launch was bound by review-claim SHA-256
`3d406677036c1d769cdd6bffde6373d5f613dc2189e2a3da47da2be96429717e`.
This producer result has `positive_claim_authority=false`. It kills only this
frozen DH384 basis/codebook. It does not kill coarse-derived graph lifting,
label-flexible algebraic codes, posterior reconstruction, or a block-level
polar refinement search.

Publication hashes:

- `CONTINUOUS_GATE.json`:
  `8492c3211bb0a8eceb8b84dd05f4c36958241e4a89d8c99f56ad2ca74b32fb87`
- `RESULT.json`:
  `f08dc7140e6f7d4889e0b8ae04bc2e40198394e3fcc52ba05f905e09e6df9263`
- terminal `COMPLETE.json`:
  `2e2382c661d984fea2fecb6cf7919d3d52246431fec218536e756615bf84746a`
