# Existing API feasibility audit

## Available and sufficient

`strata_v2_codec.emit_and_lock.build_staging` is the exact source-to-staging
implementation used by the released codec. Its numerical input is a list of
authenticated BF16 matrix metadata records. The high-level lock validator is
Qwen-specific, but the transform primitive itself is not: source paths,
shapes, roles/axes, hashes and canonical ordinals are sufficient.

`strata_expert_local_codec.run_and_pack` validates a sealed 15-block plan,
calls the unchanged `strata_v2_codec/polar_encoder.py`, verifies all three
encoder round-trip booleans, and constructs the literal current artifact. It
does not require original Qwen weights or Qwen tensor names.

`research/unifilar_wfa_entropy_census_stage0_v8/strata_sc_adapter.py` parses
that artifact, regenerates the exact selected SC decisions and reconstruction,
and can therefore prove that a produced control enters the same downstream
aperture.

## New bridge code required

The current expert-local independent scorer's numeric decoder is reusable,
but its source-binding wrapper reconstructs a literal
`model.layers.*.experts.*` tensor string. A universal scorer must instead bind
canonical slot, role, shape and retained BF16 digest, then prove that its
reconstruction digest equals the exact v8 adapter.

The v8 control consumer must not be reused unchanged. It compares a
source-derived structural digest and lacks generated BF16 source bytes. A new
v9 bridge must be separately audited; changing that comparison inside v8 would
be an unaudited weakening.

## Conclusion

The source-to-current-artifact producer is mechanically closed. The honest
control experiment remains blocked only at external evidence and consumer ABI
boundaries, not at the quantizer or polar encoder.
