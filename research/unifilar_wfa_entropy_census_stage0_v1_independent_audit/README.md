# Independent source-only hostile audit: UWFA census v1

This directory is independent of the sealed producer package
`research/unifilar_wfa_entropy_census_stage0_v1`.

The audit is bound to producer `SOURCE_MANIFEST.json` SHA-256
`1dbea65550d879c3cc6ca81974223d251d669c15f5af17fa9681800cf03cf9ff`.
It authenticates the complete producer closure before importing project code
and uses deterministic synthetic fixtures only. It must not discover, stat,
hash, enumerate, or open any Qwen/model payload, current finite artifact,
extracted selected-bit stream, or Gaussian-control payload.

The audit covers:

- all 150 exact unifilar transitions, resets, and the reset-before-emission
  `t=0` law;
- independent Q0.16 Jeffreys rounding, model serialization, exhaustive small
  arithmetic-code equivalence, causal decode, and canonical re-encode;
- every cell's CPU/CuPy count and exact arithmetic-length agreement;
- nested whole-layer/whole-expert holdout and identity-key exclusion;
- synthetic baseline replay, control ordering, and status precedence;
- model bytes, headers, directory, rate floor, page reads, cold-read formula,
  physical saving, and unchanged-reconstruction `F` conversion;
- wrong-token behavior, symlink handling, source authentication lifetime, and
  measured fixed-cell runtime scaling;
- the precise negative-result scope: current selected SC arithmetic decisions
  only, never arbitrary MPS/MERA/source-coordinate label copulas.

The final adjudication and audit manifest are generated only after the sealed
test output is replayed.
