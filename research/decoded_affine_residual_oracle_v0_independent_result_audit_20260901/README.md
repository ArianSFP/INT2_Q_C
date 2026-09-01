# Independent decoded-affine residual result audit

Verdict: **numerical favorable-family kill verified, with an unsealed pre-execution provenance limitation.**

The result is pinned at SHA-256
`e8cff5c3ee45dc209c4b4d5368e060953e4ccc8e08047a28bddca8e36e19de35`.
The producer result now resides outside the source directory at
`research/decoded_affine_residual_oracle_v0_runpod_result_20260901/result.json`.
The currently available script hash exactly matches the hash recorded by the
result. However, the producer directory contained no pre-execution sealed source
manifest or receipt. This audit can establish numerical self-consistency between
the downloaded result and current script, but cannot authenticate that the
script, topology, constants, or claim boundary were frozen before execution.

Using only reported matrix rows, the verifier recomputes:

- baseline SSE `500.39553685426534` and source energy `16192.89450885593`;
- relative MSE `0.030902167403153148` and 2.5-bpw baseline F
  `0.9888693569009007`;
- all 12 scale/bias/affine width-cell SSE sums and the row-plus-column cell;
- every coefficient bpw, fraction of baseline SSE, and favorable transfer F;
- ordering, best-cell selection, and all pass/fail decisions.

The best cell is the width-2048 scale oracle: exact source-fit FP64 SSE
`499.9692952474626`, fraction `0.9991481906303915`, nominal coefficient rate
`0.0078125` bpw, and favorable transfer F `0.998785937659889`. Every one of the
13 favorable cells remains above F=0.8, confirming
`HARD_KILL_AFFINE_CORRECTION_FAMILY` conditional on the copied script/result
being the intended experiment.

This is deliberately a favorable envelope: exact FP64 source-fitted coefficients
are granted while only nominal FP16 bits are charged, and the observed correction
fraction is assumed to transfer unchanged to a lower-rate coarse stream. The
kill is specific to these affine/additive correction families, not nonlinear
corrections or a universal codec bound.

No Qwen sources, decoded checkpoint payload, fresh validation, CuPy, or GPU were
opened by this result-only audit.

Verify with:

```text
python -B verify_audit.py --audit-dir .
```
