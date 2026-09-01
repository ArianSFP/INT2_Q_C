# Independent same-layer expert-alignment result audit

Verdict: **PASS with a selection-receipt limitation — narrow Up/Down ancestry kill confirmed.**

The result is pinned at SHA-256
`a1ef6fb136027525b6312635cdcca320f05f51c1340c3875b32192454aac1bb3`
and now resides outside the source closure at
`research/same_layer_expert_alignment_superoracle_v0_runpod_result_20260901/result.json`.
The audit holds exact copies of the result, sealed design, runner, and source
manifest. It did not open any of the 32 Qwen auxiliary payloads, pinned data,
fresh validation, CuPy, or GPU work.

All 32 source receipts form the exact Cartesian mapping of 16 frozen layer-15
experts with roles Up then Down. Their expert/role identities, paths, sizes, and
hash syntax are structurally valid. The 32 scored rows use the same order.

Independent arithmetic gives:

- captured energy `412.51977756375265`;
- source energy `26554.382796073027`;
- pooled capture `0.015534903625203362`;
- Up-only capture `0.012359993622486992`;
- Down-only capture `0.018643001979871822`;
- absolute cushion `0.001` and favorable capture `0.016534903625203354`;
- required capture `1 - 2^(-2*0.11356063457) = 0.14566207552117194`;
- shortfall `0.1291271718959686`.

This confirms `HARD_KILL_SAME_LAYER_UP_DOWN_ANCESTRY`: even the illegal
many-to-one, role-independent, source-fitted oracle with free references,
mappings, coefficients, reads, and the extra capture cushion cannot close the
existing composite gap as the sole missing module.

The sealed source constructs references from all 15 non-target experts, maps
each selected index to `(reference_expert, row)` correctly, and hashes 768
little-endian int64 pairs per target/role. However, the result emits only those
32 hashes—not the selected pairs. Their structure and uniqueness verify, but
the exact selected-expert mappings cannot be replayed from the result alone
without reopening payloads. This is a transparency limitation, not an arithmetic
contradiction.

Gate ancestry, nonlinear/shared generative decoders, and activation-aware
functional compression remain outside the kill.

Verify with:

```text
python -B verify_audit.py --audit-dir .
```
