# FUSEED-U32 source-free complete scan probe v0

This unsealed engineering probe actually traverses every `uint32` seed from
zero through `2^32-1`, rather than extrapolating from a smaller domain. It
uses the hash-bound synthetic 80-`normal4` kernel from the throughput probe,
keeps an exact tie-aware Top-K in each shard, merges the exact global Top-K,
and repeats the complete traversal to check determinism.

It has no model/source argument and makes no initializer or retention claim.
The synthetic two-role score is deliberately much cheaper than the proposed
multi-domain FUSEED objective. Exact producer ABI parity, 33-domain controls,
state journaling, selection/validation separation, Qwen capture, and a
physical residual codec remain outside this probe.

## Measured result

On the pinned RTX 5090 with CuPy 14.2 and `CUDA_VISIBLE_DEVICES=0`, both
complete traversals covered exactly `256 * 2^24 = 2^32` seeds. Their exact
per-shard and global Top-K seed/value hashes were identical. Median full-scan
wall time was `2.9155251910560764 s`: `1.9475334925809875 s` summed kernel
time, `0.8233270378550515 s` exact finite/tie-aware shard selection, and
`0.13585911208065227 s` host merging. The maximum Top-K boundary-tie
cardinality was one in both traversals.

This removes linear-extrapolation and 32-bit wrap uncertainty for the
synthetic kernel only. It does not project that timing onto the eventual
33-domain statistic; that objective must receive its own frozen calibration.

Run from an absent output parent:

```sh
env CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B -I \
  complete_scan.py --output ABSENT_PARENT/result.json \
  --shard-candidates 16777216 --top-k 2048 --repetitions 2
```

`verify_result.ps1` independently checks source bindings, exact domain
coverage, repeat hashes, medians, and zero model/payload access. Digests are
listed in `artifact_sha256.txt`, which excludes its own digest.
