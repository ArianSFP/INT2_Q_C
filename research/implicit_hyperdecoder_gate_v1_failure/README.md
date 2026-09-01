# SILWARP auxiliary v1 pre-training failure

This directory preserves the complete v1 process evidence.  The authorized
auxiliary run authenticated the frozen fit and calibration identities, decoded
54 complete fit records, and then aborted before its first training update
while constructing `null_a` for fit record `L31/E31/down`.  It produced no
checkpoint, calibration score, confirmation access, model, or result.

The cause was representation-only.  The Qwen source's exact mean
`-4.649028828573876e-09` rounded to FP16 negative zero (`0x8000`), while the
mathematically zero control mean `2.238990423908867e-19` rounded to positive
zero (`0x0000`).  Their numeric means and RMS metadata were equal, but v1's
intentional bitwise metadata assertion rejected the distinct IEEE zero
encodings.

SILWARP v2 makes the minimal deterministic repair: immediately after
nearest-even FP16 conversion, either zero mean is canonically represented as
`+0`.  This changes no numeric value, normalization, control realization,
architecture, hyperparameter, split, seed, score, or threshold.  The exact
bitwise assertion is retained after canonicalization.

Artifact identities and the precise access boundary are recorded in
`failure_receipt.json`.  The preserved log and GPU preflight receipt must not be
edited.
