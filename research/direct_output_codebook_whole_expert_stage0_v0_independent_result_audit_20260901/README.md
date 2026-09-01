# Independent direct-output d8 result audit

Verdict: **PASS — narrow `K=32768`, `d=8` direct-output cell kill, without collapse.**

The audit holds the exact sealed source evidence, downloaded result, and all
eight emitted binaries. It used only the standard library and did not open Qwen
sources, fresh validation, or run GPU work.

The result is pinned at SHA-256
`701e3ebe3da89c4fa5e72bca335a1b33e7bb57937851956ee139630d26b9b991`.
The complete producer result directory now resides outside the sealed source
closure at `research/direct_output_codebook_whole_expert_stage0_v0_runpod_result_20260901`.
The result's canonical internal lock, source manifest, source-lock identity,
18 source receipts, physical-rate/read arithmetic, and hypothetical 128-expert
arithmetic-only ledger all verify.

Each of the four global files is exactly 528,384 bytes: a 4,096-byte header and
a finite little-endian FP16 `32768 x 8` table. The fixed binary prefix and JSON
metadata bind the seed, source/control identity, code count, dimension, and
15-bit indices; remaining header bytes are zero. Each of the four moment files
is exactly 55,296 bytes (`18 x 768 x 2` FP16 values), finite, identical, and its
18 slices match the reported per-matrix hashes.

The recomputed six-expert fixed prefix is 2.0400390625 bpw and the favorable
first-stage requirement is `q <= 0.047300320854109984`. Results are:

| Seed | Source q | Source F | Source s | Gaussian F | Matched advantage s |
|---:|---:|---:|---:|---:|---:|
| 2026090111 | 0.106009596 | 1.792961970 | -0.421172444 | 1.781126001 | -0.004777654 |
| 2026090112 | 0.105958749 | 1.792101996 | -0.420826375 | 1.781199753 | -0.004401716 |

The verifier recomputes matrix, expert, and pooled `q`, then every `F`, `s`, and
matched advantage. Both source `F` values exceed 0.8. All five fixed checkpoints
pass for both source seeds and both Gaussian controls, and every collapse-reason
list is empty. Thus the kill is representational for this exact frozen d8 cell,
not an accidental collapse kill.

Scope is intentionally narrow: this does not reject arbitrary K-means, other
vector dimensions or codebooks, additive/nonlinear VQ, or SoftBinary Coding.
The residual stage is an ideal Gaussian oracle, not a finite codec or model-wide
result.

Verify with:

```text
python -B verify_audit.py --audit-dir .
```
