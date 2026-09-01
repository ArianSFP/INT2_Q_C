# FUSEED-U32 source-free throughput probe v0

This unsealed engineering microbenchmark tests only whether a CuPy kernel can
derive a base seed from its shard ordinal, execute 80 `curand_normal4` bundles
(320 values), reduce them to one two-role score, and retain an exact shard
Top-K without materializing candidate anchors. It has no model/source argument
or network path and makes no retention, initializer, or codec claim.

The linear full-`u32` projection is a planning measurement, not launch
evidence. A real implementation must calibrate exact ABI coordinate maps,
sharding, global merges, controls, journals, numerical parity, and modeled
retention under a separately frozen protocol.

## Measured result

The corrected script (SHA-256
`001a3d08902441ee47501ff5a99bb0ce5159bff35b23b5cd11491539a564f401`)
ran on the pinned RTX 5090/CuPy 14.2 runtime with
`CUDA_VISIBLE_DEVICES=0`. Three repetitions over `2^24` candidates produced
identical Top-K seed and value hashes. The median kernel time was
`0.007374175009317696 s`; its strictly linear `2^32` projection is
`1.8877888023853302 s`. Median warm end-to-end time projects to
`2.443514108657837 s`. The first Top-K call incurred about `0.194 s` of cold
allocation/JIT overhead that the warm projection does not include.

This establishes only that a fused, non-materializing exhaustive screen is
computationally plausible. The measured score is a synthetic two-role proxy,
not the proposed 33-domain FUSEED score. The run does not cover exact ABI
parity, real shard traversal, global Top-K merging, checkpoint journals,
selection controls, Qwen weights, or MSE retention. Those remain mandatory
before scientific use.

## Attempt log

The first remote attempt used script SHA-256
`5ac401f5379d899e3ade1bbd4ea64d2a130e4782dd36c6b0b28de5cfa964fbe5`
and stopped before writing an output directory: CuPy 14.2 rejected the
NumPy-style tuple passed to `lexsort`. The only code change for the next
attempt stacks the two sort keys into the 2-D array required by CuPy. Neither
attempt exposes a model or payload path.

## Reproduction and verification

Run on the pinned worker from an absent output parent:

```sh
env CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B -I \
  benchmark.py --output ABSENT_PARENT/result.json \
  --candidates 16777216 --repetitions 3 --top-k 2048
```

From PowerShell, `./verify_result.ps1` checks the script binding, zero-access
claim, row count, repeat determinism, medians, and linear projections. File
digests are in `artifact_sha256.txt`; the manifest deliberately excludes its
own digest.
