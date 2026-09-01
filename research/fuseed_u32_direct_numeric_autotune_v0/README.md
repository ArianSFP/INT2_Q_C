# FUSEED direct-counter FP32 screen autotune v0

This source-free CuPy probe keeps the exact direct Philox/Box--Muller/BF16
generator and changes only the per-domain `x*w` accumulators to FP32. It tests
four, eight, and sixteen warps/block plus two register caps, requiring all
variants to emit identical q sentinel bytes. The winner is selected solely by
median time over five predeclared repetitions.

The 800-second planning margin is deliberately below FUSEED-v1's 900-second
cap so exact FP64 refinement and bookkeeping have room. Passing it is only a
performance promotion signal; a distinct cascade still needs numerical and
modeled-retention audits before any Qwen access.

## Frozen outcome

The best of the five prospectively listed launch shapes was
`warp8_block256_r80`, at `0.07442336296662688` seconds for `2^20`
candidates.  Its exact linear projection for three complete u32 ABI scans is
`914.5142841339111` seconds.  This is above the unchanged 800-second margin,
so the FP32-screen branch is an early runtime kill and received no Qwen/model
access.  All five launch variants produced the same q sentinel hash.
