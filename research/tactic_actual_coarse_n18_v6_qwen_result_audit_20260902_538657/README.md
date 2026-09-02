# TACTIC actual-coarse N18 v6 Qwen result audit

Date: 2026-09-02

The corrected independent v1 replay passed. It did not trust the producer's
reported score: it independently parsed and causally decoded all 18 streams,
verified little-endian I32 inverse state, reproduced every canonical symbol,
re-encoded the complete `COARSE.bin`, and rescored the externally pinned BF16
Gate, Up, and transposed-Down inputs in original source coordinates.

## Audited result

- Physical rate: exactly `307/128 = 2.3984375 bpw`.
- Pooled relative MSE: `0.036975150060595235`.
- Coarse diagnostic `F = D * 2^(2R)`: `1.0278108682335156`.
- Pooled SSE: `89.64596664948934`.
- Pooled source energy: `2424.492300979892`.
- Frame: `1,414,656` bytes.
- `COARSE.bin` SHA-256:
  `6c13780bf1494567f91bc73bf6afd8846c6e3326cac329e4d8e3faf48a9051d7`.
- v6 terminal `COMPLETE.json` SHA-256:
  `6b5e96c42518a29493e68237d649daad2e25f44a509ce7535425f83fd79fbb37`.
- Input manifest SHA-256:
  `6f6a0f174cd5b9c2b52ef29efd612e4520ef77afa6cc950ebec8c7e055fedcaa`.
- Independent audit receipt SHA-256:
  `576d11058a77e77826140686d6eed0c6d2782e83c7338363e568a79a0f43f31b`.

The residual fine stage at literal 2.5 bpw must therefore remove
`1 - 0.025 / D0 = 0.32387022205373717`, or 32.3870222% of this coarse SSE.

## Failure history

Two earlier audit attempts failed closed and remain preserved:

1. The first external pin used the same decoder bytes at a noncanonical path.
2. Auditor v0 omitted the producer-required authenticated field
   `scorer_uses_exact_encoder_input_bytes: true` from its expected exact score
   mapping. The producer receipts were correct; v1 adds hostile tests for
   missing, false, and extra score-attestation fields.

The passing auditor source manifest is
`5386571db2a8e828c09368f603b3ccf0ccf3936204e7e06231d5c5798eb9f97f`.
The external-pin file SHA-256 is
`c19e0b00c83e4c2ad49cc23244475a0e23a6c7701475d4ce1d462d1b557b3bfe`.

## Claim boundary

This authenticates one externally pinned 768x2048 three-role coarse pilot.
It is not a final 2.5-bpw codec, a universal SwiGLU-MoE result, an inference
HBM measurement, or evidence for `F <= 0.8`.
