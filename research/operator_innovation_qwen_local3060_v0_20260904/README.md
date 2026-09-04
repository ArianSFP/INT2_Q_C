# Operator-conditioned innovation probe v0

Status: exploratory Qwen aperture on the pinned local RTX 3060.  This is not a
finite codec and cannot establish the final `F <= 0.8` claim.

This probe tests the part of the operator-innovation proposal that is not
contained by the existing same-coordinate semantic predictor or decoded-SVD
spectral shrinker.  It causally regenerates the exact audited 2.5-bpw STRATA
reconstruction and projects its true source residual onto nested banks of
decoder-visible whole-matrix features.

For aligned expert matrices `G`, `U`, and `V=D.T`, the banks include:

* constant and scalar-rescaling corrections;
* unary cubic `Y_r Y_r.T Y_r` corrections;
* mixed cubic products `Y_a Y_b.T Y_c` for all 27 ordered triples;
* the symmetry-aware Gram--Hadamard feature
  `((Y_a Y_a.T) * (Y_b Y_b.T)) Y_r` proposed for each target role.

Every matrix first receives its own exact source-fitted coefficients.  This is
a deliberately favourable upper aperture.  The program also reports
leave-one-expert-out coefficients, fitted on five experts and evaluated on the
sixth, so source-fitted and portable evidence cannot be confused.

The early-kill rule is:

* stop the tested compact operator span if its favourable source-fitted
  improvement is below 10% of baseline residual SSE;
* promote to alignment-destruction and matched-control work only at or above
  10%;
* it can be a direct target candidate only if the fully charged transfer reaches
  `F <= 0.8` (about 20% capture after even a small parameter charge).

The calculation uses CuPy for the matrix products and projection reductions.
The compact checkpoint intentionally omitted its 226 MB decoded scratch, so the
program reconstructs that scratch from the checked-in physical container and
requires its historical SHA-256 identity before fitting.

## Qwen result

The exact six-expert run is a hard kill for this compact operator span:

| Bank | Source-fitted capture | Leave-one-expert-out capture | Favourable F |
|---|---:|---:|---:|
| Scalar bias/scale | 0.00580% | 0.000012% | 0.988840 |
| Unary cubic | 0.23280% | 0.19448% | 0.986609 |
| Symmetry-aware mixed | 0.23896% | 0.19566% | 0.986562 |
| All 27 cubic products | 0.34640% | 0.29766% | **0.985847** |
| All cubic plus proposed degree-five feature | **0.34754%** | 0.29753% | 0.985850 |

The proposed symmetry-aware terms add only about `0.00616` percentage points
of capture beyond the unary cubic bank.  The entire unrestricted cubic span
adds only about `0.11460` percentage points beyond unary.  All 30 features are
retained in every full-bank solve; retained condition numbers are 112--644, so
the negative is not caused by rank truncation.

The most favourable bank still leaves `F=0.98585`, versus the required
`F<=0.8`.  It has less than one twenty-eighth of the 10% control-launch gate
and less than one fifty-fifth of the approximate direct target capture.
Consequently no matched-control run, Bayesian denoiser, or closed-loop
innovation packet is warranted for this feature family.

The prescribed blockwise follow-up was also executed.  Granting a separate
exact coefficient vector to every 32-row block raises raw capture only from
`0.34754%` to `0.40013%`.  Its charged coefficient rate rises to
`0.00732422 bpw`, so favourable `F` worsens to `0.994964`; the global
all-cubic bank remains the best rate-distortion cell.  Widths 128 and 64 are
likewise dominated.  This closes the proposal's named 32/64/128-neuron
scales without spending compute on controls that cannot rescue the dominant
upper aperture.

This result kills only compact global scalar combinations of the tested
degree-three operator words and the named degree-five Gram--Hadamard term.  It
is not a converse for spatially varying operator coefficients, a large neural
operator, activation-weighted functional loss, or arbitrary Volterra models.

Authoritative exploratory result:

* `RESULT.json`
* SHA-256 `ae63d7a0dd931515eb64cefee3f30873c63100b179cb33084c78a374f4167a7b`
* `RESULT_BLOCKWISE.json`
* SHA-256 `165b9e213867b36ee16d34f37b84a689c172d29a79d55985162420c40f49d3fd`
* decoded reconstruction SHA-256
  `af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0`

Run the dependency-free consistency verifier with:

```powershell
& C:\INT2__compression\.tools\python\cpython-3.12.14-windows-x86_64-none\python.exe `
  -I -B research\operator_innovation_qwen_local3060_v0_20260904\verify_result.py
```

Example:

```powershell
& C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  research\operator_innovation_qwen_local3060_v0_20260904\run_probe.py `
  --release results\qwen\strata_expert_affine_checkpoint `
  --source-root research\atlas_sigma\dataset\materialized\qwen_six `
  --scratch tmp\operator_innovation_decoded_v0 `
  --output research\operator_innovation_qwen_local3060_v0_20260904\RESULT.json
```
