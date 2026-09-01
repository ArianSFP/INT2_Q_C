# FUSEED exact direct domain-collapse probe v0

This source-free CuPy probe measures a distinct FUSEED-v2 compute idea: scan
the exact source objective exhaustively while moving the 32 descriptive
control objectives out of the exhaustive inner loop.  The direct Philox
counter, hash-bound cuRAND Box--Muller implementation, FP32 role scaling,
BF16 rounding, common moments, per-domain cross-moments, decoded-FP16 affine,
and final score all remain unchanged; in particular, every retained source
calculation remains FP64 where v1 used FP64.

The probe prospectively tests active-domain counts `1,2,4,8,16,33` under four
launch/register configurations, with five timings each.  Every domain's
sampled q bytes must be identical across every configuration that contains
that domain.  An exact source-only kernel projection below 800 seconds is
only permission to construct a hardened full-shard calibration.  It is not a
Qwen result, a control/significance argument, or authorization to open model
payloads.

## Frozen outcome

With all three v1 ABI families retained, the exact one-domain winner was
`block256` at `0.12515333795454353` seconds for `2^20` candidates, projecting
to `1537.884216785431` seconds.  Domain collapse therefore fails the fixed
800-second three-ABI margin and is stopped without Qwen/model access.  The
same prospective timing implies `512.6280722618103` seconds for one full u32
ABI, motivating a *distinct* single-ABI design rather than relabeling this
failed result.  Every common domain's sampled q bytes were identical across
all containing launch variants.
