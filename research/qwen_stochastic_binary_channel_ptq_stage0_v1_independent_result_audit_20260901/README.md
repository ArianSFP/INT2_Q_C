# Independent QSB-PTQ-v1 result audit

Verdict: **PASS — all three cells are valid favorable-oracle hard kills; matched controls correctly did not run.**

This result-only audit pins the downloaded result at SHA-256
`c9b8d26e8d09bbb225288e7787d9ef0154a6a958cc4cf1798cd2d3358b3eeae2`
and the pre-execution source manifest at
`aaae42924d01f3508bd66ad8efa586549a687e69fe6a8f31bdcc9d1c8806707f`.
The producer result now resides outside the exact source closure at
`research/qwen_stochastic_binary_channel_ptq_stage0_v1_runpod_result_20260901/result.json`.
It copied and held the exact result, design, panel bindings, runner, and source
manifest. It did not open Qwen payloads, fresh validation, or run CuPy/GPU work.

All six expert KL values in every cell are below the separate 97% execution
limit and the true physical reservoir. The smallest margins are:

| Cell | Minimum margin below 97% | Minimum physical-reservoir margin |
|---|---:|---:|
| QSB215 | 48,483.259732 bits | 351,259.579732 bits |
| QSB230 | 52,400.100910 bits | 376,803.300910 bits |
| QSB250 | 57,727.995580 bits | 409,656.315580 bits |

The audit independently recomputes every matrix capture, all six expert folds,
all three role folds, pooled capture, delete-expert jackknife statistics, exact
required capture, and `F = relative_mse * 2^(2R)`:

| Cell | Pooled capture | Upper 3-SE | Required capture | Recomputed F |
|---|---:|---:|---:|---:|
| QSB215 | 0.733695344 | 0.735119697 | 0.959543471 | 5.265991211 |
| QSB230 | 0.740732191 | 0.742170097 | 0.967265386 | 6.336236164 |
| QSB250 | 0.745510752 | 0.746963173 | 0.975000000 | 8.143655928 |

Thus each favorable upper-three-SE capture remains far below its exact target.
The runner correctly records
`HARD_KILL_FAVOURABLE_ORACLE_UCB_BELOW_EXACT_REQUIREMENT`, skips all Gaussian
controls, and returns `POLICY_REJECT_ALL_RATE_CELLS`.

The claim remains narrow: this is an adapted, source-fitted, source-leaking
topology bound on the reused panel. It is not independent held-out evidence,
serialized channel simulation, operational compression, Gaussian/TCQ evidence,
a Shannon-limit result, or fresh validation.

Verify with standard Python:

```text
python -B verify_audit.py --audit-dir .
```
