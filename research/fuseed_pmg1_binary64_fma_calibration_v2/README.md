# FUSEED-PMG1 explicit-FMA binary64 calibration v2

The frozen binary64-v1 run projected to `696.8922319519334` seconds and
therefore failed its prospectively fixed 650-second stage-0 margin.  This
successor does not relax that limit.  It uses a distinct arithmetic
architecture already allowed by the PMG1 design contract: every dominant
FP64 multiply-add is written explicitly as correctly rounded `__fma_rn`,
while centered moments and remaining products/divisions use explicit
round-to-nearest intrinsics.  NVRTC contraction is enabled for the unchanged
float32 cuRAND generator path, restoring its native compiled shape without
making the FP64 objective's contraction sites implicit.

The wrapper hash-binds the entire frozen binary64-v1 implementation template,
checks every source rewrite cardinality, gives the derived run a distinct
schema/journal status, and retains the same 650-second margin, binary64 metric,
packed 12-byte record wire, plan, parity, and no-model-access boundary.

## Frozen outcome

The three deterministic full-shard replays passed.  Median complete shard
time was `2.0215128749841824` seconds and the full-u32 stage-0 projection,
including finite/canonical-zero validation, exact Top-K, fsynced packed
journals, one-time cold excess, and the global merge shape probe, was
`520.8358833260136` seconds.  This passes the unchanged 650-second margin and
leaves `379.1641166739864` seconds inside the full 900-second wall limit.

The executed performance cubin SHA-256 is
`41e71c07819ac6ce99e0bfb4c3903aa8400e20fa955ce3157e215a7d732b55ac`;
the result SHA-256 is
`82e29cbfc8ec1ac23761c37712a3fda3d2745b04c9a71ae296ce864796ddc75e`.
All direct/shifted/sequential/Torch parity panels replayed three times, but the
result remains source-free and pending independent audit plus separate exact
stage1/stage2/validation timing before any Qwen access.
