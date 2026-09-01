# FUSEED-PMG1 binary64 stage-0 calibration v1

This distinct source-free successor corrects the mismatch discovered in the
otherwise successful v0 calibration.  The frozen PMG1-v2 design requires the
final stage-0 metric itself to be finite IEEE binary64 capture, ordered
descending with unsigned seed as the tie-break, and each journal record to be
packed as `u32 + binary64` (12 bytes).  V0 used v1's binary32 error-ratio wire
and therefore cannot authorize the design despite its 516.67-second timing.

V1 keeps the exact direct Philox/cuRAND/BF16 generator and FP64 moment path,
changes only the final metric/wire, disables contraction with an explicitly
bound NVRTC option list, and replays all hardened direct, original-offset
sequential, and PyTorch state/value parity gates three times.  It also proves
that its execution plan hashes to the design draft's independently serialized
ABI1 bundle digest.

The prospective stage-0 margin is 650 seconds, reserving 250 seconds inside
the unchanged 900-second full-pipeline limit for separately measured stage1,
stage2, validation, and final journaling.  Passing this probe alone does not
authorize Qwen/model access.

## Frozen outcome

The three deterministic full-shard replays produced a median complete shard
time of `2.7174683440243825` seconds.  Including one-time cold excess and the
global merge shape probe, the complete-u32 projection is
`696.8922319519334` seconds.  This exceeds the prospectively frozen
650-second stage-0 margin, so binary64-v1 is an early kill without Qwen/model
access.  Its result SHA-256 is
`2e2e6dc73a16921221cfd309a243fce6794464c201f80714ca3e2dcc94078c4d`.
The result remains useful as the no-contraction control for the distinct
explicit-FMA successor; its gate is not relaxed.
