# TETRAPATH-BA label-flexible Qwen upper bound

This probe replaces the non-convex hard-label iteration with a batched
Blahut--Arimoto rate-distortion relaxation.  For each selected 2,048-weight
block it compares four independent four-level channels with one unrestricted
256-reconstruction four-way channel.  The reproduction levels and source MSE
are identical.  The full channel is stochastic, its per-block distribution is
free, and its time sharing is free, so it is strictly more favourable than a
finite deterministic entropy code.

That makes a null result useful as an early upper-bound screen.  A positive
result is not a codec and must survive held-out modelling, literal bytes and
the routed-read ledger.  Independently permuting the four variables inside
each block supplies a finite-sample/marginal-preserving control.

All large distortion and posterior operations execute through CuPy on the
pinned local RTX 3060.  The initial aperture is intentionally small and can be
expanded only if it survives.

## Qwen result

The final 1/64-block aperture covered all eight fixed expert pairs in the
authenticated layer-15 panel.  Unrestricted full-vs-independent gain averaged
`0.062354299270005464 bpw`; after a within-block independent-permutation
control it averaged `0.028618142219119652 bpw`.  One pair reached
`0.04600639091553179 bpw` after that control.

That apparent survivor was entirely pair-factorizable.  After replacing the
independent baseline with the lower hull of all three 2+2 factorizations, the
best control-corrected irreducible four-way result was
`-0.0002816836169618857 bpw`, and the eight-pair mean was
`-0.0018706304220411875 bpw`.  A deeper 240-iteration rerun of the original
outlier remained negative (`-0.0018132283432135375 bpw`).

Disposition: hard-kill irreducible coordinate-memoryless four-way synergy at
this four-level aperture.  This does not close pairwise flexible coupling,
multiscale RENORM-Q, cochain factors, or the real six-plane STRATA-RM6 search.

Authoritative result SHA-256:
`3b68c4ee7115bfb8d5f6b6e8027a2bb27c5c0f6d358647b4c4394182e7158353`.
