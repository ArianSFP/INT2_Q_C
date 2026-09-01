# FUSEED-U32 source-free numerical cascade probe v0

This unsealed probe compares the exact FP64-cross-moment FUSEED shard metric
with a fast first stage that changes only the four per-domain `x*w` sums to
FP32. Common moments, the affine solve, FP16 alpha/mu round-trip, and final q
evaluation remain FP64. It measures kernel speed and how much of exact
Top-8192 survives deterministic approximate Top-M shortlists.

This is empirical synthetic evidence, not a numerical interval proof or a
modeled-retention certificate. It cannot authorize Qwen access. Any promoted
cascade must be a distinct v2 protocol with prospectively frozen shortlist,
exact FP64 refinement, control symmetry, and conservative retention gates.
