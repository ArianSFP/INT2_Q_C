# TETRAPATH-4 fixed-label Qwen probe (local RTX 3060)

This is a deliberately bounded exploratory measurement.  It asks whether the
nearest four-level Up/Down labels of pairs of same-layer Qwen experts contain
four-way entropy that is not explained by independent, best 2+2, Chow--Liu,
or pairwise-maximum-entropy models.

It does **not** test label-flexible TETRAPATH, a finite packet, STRATA's real
six-plane code, or a universal held-out model.  A null result closes only the
fixed-label memoryless census.  A positive result is only a promotion signal.

The runner authenticates the existing layer-15 panel, uses CuPy on the pinned
local RTX 3060 for nearest-label selection and joint tuple counts, and scores
the canonical CPU diagnostics from the source-only TETRAPATH implementation.
It also repeats the census after independently permuting each variable, which
preserves every marginal while destroying aligned cross-variable structure.

Across all eight fixed expert pairs, maximum raw four-way gain over the best
independent/2+2/tree model was `0.00003870032355246522 bpw`; maximum
control-corrected gain was `0.000013698442561604907 bpw`.  Maximum
control-corrected connected information beyond the pairwise maximum-entropy
law was `0.000004263832382944699 bpw`.  This is a decisive fixed-label
memoryless hard kill against the `0.045 bpw` continuation gate.

Result SHA-256:
`2dab5c4175149d92f46409724bc2d204e05452e5072efcbb23cffad9a1f19418`.
