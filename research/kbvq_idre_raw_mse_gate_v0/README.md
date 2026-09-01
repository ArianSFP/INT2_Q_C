# KBVQ-IDRE raw-MSE gate v0

## Outcome

This source package identifies a real architectural distinction in
KBVQ-MoE's IDRE, but finds no viable raw-MSE cell under the currently available
aggregate evidence.

The bounded, physically allocated result is:

| quantity | best positive-rank cell | target |
|---|---:|---:|
| mode / rank / cap | role-specific / 8 / 2.50 bpw | any 2.15–2.5 bpw |
| actual physical rate | 2.5 bpw | <= 2.5 bpw |
| FP16 factors + all headers | 0.065185546875 bpw | charged |
| residual payload | 2.434814453125 bpw | — |
| exact two-role Gaussian-waterfill relative MSE | 0.033404428110059906 | — |
| `F = MSE * 2^(2R)` | **1.068941699521917** | **<= 0.8** |
| `s = -0.5 log2(F)` | **-0.048091585039837736 bpw** | **>= 0.16096404744368115** |
| worst exact 4-KiB cold read | 1.075x | < 2x |

The cell is therefore a **hard kill under the declared two-role
diagonal-Gaussian residual model**.  It is not yet a per-mode residual converse.
The coordinated CuPy replay in `stage0_gate.py` is frozen to compute that finer
per-mode waterfill if root later authorizes a payload replay.  No GPU job and no
new numeric Qwen read were performed while preparing this package.

## What the primary paper actually does

The only method sources used here are the
[ICLR 2026 camera-ready paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/5f868c3e0050f13ec82d3694df531de1-Paper-Conference.pdf)
and the [author repository](https://github.com/xuzukang/kbvq_moe).  At the seal
date the author repository exposed a README and the paper PDF, but no executable
implementation, so the gate follows Sections 4.1 and A.2–A.3 of the paper.

For each weight role, IDRE maps weights through an activation-coherence basis,
stacks all experts on the output-row axis, and takes a leading *right* singular
subspace.  It stores:

```text
shared per layer/role:  B_r in FP16, shape [2048, k]
private per expert:     A_(e,r) in FP16, shape [768, k]
decoded component:      L_(e,r) = A_(e,r) B_r^T
quantized target:       W_(e,r) - L_(e,r)
```

The paper proves its activation-output objective as

```text
Tr(E C_X E^T) = ||E U_X||_F^2 .
```

Our target is instead unweighted raw weight MSE.  Its exact specialization is
`C_X = I`, hence `U_X = I`; the relevant IDRE basis is the leading right
singular subspace of the stacked raw canonical weights.  The auxiliary cache
contains weights, not an authenticated activation covariance.  BCOS is excluded
because it corrects activation-output moments and is not raw-weight evidence.

## Why this is not the already-killed whole-expert PCA

The nearby experiments must not be conflated:

| artifact | representation | why it differs |
|---|---|---|
| `neural_flow_oracle/shared_expert_basis_oracle.py` | flattened 1,572,864-value matrix templates; one scalar per template | tests collinearity between whole matrices |
| `shared_subspace_gate/shared_subspace_gate.py` | shared left/right transforms; both selected and complementary bands are quantized | omits IDRE's private FP16 low-rank factors and does not subtract an exact retained component |
| `full_axis_butterfly_codec_v1/dense_klt_oracle.py` | full-rank 2048-axis transform followed by waterfill | transforms every mode; no subtractive FP16 low-rank component |
| this package | shared 2048-coordinate right basis plus 768 private coefficients per rank, expert and role | retains the factorized component and quantizes only its residual |

The distinction is genuine.  Nevertheless, the authenticated
`shared_subspace_gate/result.json` contains the exact held-out source and right
projection energies needed to score an IDRE split without reading the tensors
again.  It used the same 12-fit/4-validation split and the same joint and
role-specific leading bases.  Its result and independent PASS receipt are bound
by SHA-256 in `design_lock.json`.

## Do not confuse free capture with a physical result

The rank-256 sufficient statistics are:

| mode | exact/free capture | `q=1-capture` | gross `-0.5 log2(q)` |
|---|---:|---:|---:|
| one joint Up/Down basis | 0.17233502472544554 | 0.8276649752745544 | 0.13644059372654344 bpw |
| paper-faithful role-specific bases | 0.18422984319949592 | 0.815770156800504 | 0.14688268234588053 bpw |

These values are **gross signal diagnostics only**.  It would be misleading to
call either one `F`: removing rank changes the number of residual dimensions,
and a valid comparison must both charge the factors and reallocate the remaining
bits.  Rank 256 is also deliberately outside the <2x physical read envelope.

For every frozen candidate rank `{8,16,32,64,96,128,192,256}` and every rate
`{2.15,2.30,2.50}`, the default gate performs this allocation:

1. serialize the shared FP16 basis, expert-private FP16 factors, a 4-KiB shared
   header, and a 512-byte expert header;
2. grant the FP16 component **zero rounding error**, favoring IDRE;
3. allocate zero residual bits to its captured dimensions;
4. put every remaining frame bit into the residual;
5. solve exact Gaussian reverse-waterfill between the Up and Down residual
   bands; and
6. evaluate with byte-derived `R_actual` and exact worst-frame 4-KiB pages.

The first positive-rank rows are:

| mode | rank | cap | side bpw | payload bpw | `F` | cold pages/read |
|---|---:|---:|---:|---:|---:|---:|
| role-specific | 8 | 2.15 | 0.0651855469 | 2.0848134359 | 1.0709775780 | 1.0901167948x |
| role-specific | 8 | 2.30 | 0.0651855469 | 2.2348124186 | 1.0701045927 | 1.0824284937x |
| role-specific | 8 | 2.50 | 0.0651855469 | 2.4348144531 | **1.0689416995** | 1.075x |
| joint role | 8 | 2.15 | 0.0645345052 | 2.0854644775 | 1.0733634612 | 1.0513570866x |
| role-specific | 16 | 2.50 | 0.1289876302 | 2.3710123698 | 1.1414675435 | 1.1416666667x |
| role-specific | 64 | 2.50 | 0.5118001302 | 1.9881998698 | 1.7370350635 | 1.5375x |

Thus the full-precision private coefficient cost grows much faster than the
held-out raw-energy capture.  The paper's recommended rank ratio `1/128`
corresponds to rank 6 for a 768-rank Qwen3 matrix and is below the first frozen
rank; interpolation is not used as evidence.

The exact page ledger's maximum legal rank is:

| mode | 2.15 bpw | 2.30 bpw | 2.50 bpw |
|---|---:|---:|---:|
| one joint basis | 205 | 219 | 239 |
| two role-specific bases | 102 | 109 | 119 |

The default aggregate gate is exact for its two declared residual bands.  It
does not know how energy is distributed among individual residual right modes.
The separately interlocked CuPy replay computes all 2,048 validation-mode
energies and then performs an exact per-mode diagonal-Gaussian reverse-waterfill.
Until that replay is authorized, the narrow two-role kill must not be promoted
to a universal IDRE converse.

## No additive composite claim

The existing honest ideal composite has `s = 0.0474034129` and still needs
`0.11356063454368115` bpw.  Naively adding its `s` to the illegal free
role-specific rank-256 diagnostic gives `s = 0.19428609524588053`, or
`F = 0.76388523407141`.  That number is explicitly **invalid**:

- the rank-256 IDRE object violates the read envelope and charges no factors;
- the evidence comes from a different auxiliary panel;
- both branches may explain the same role/scale/polar source energy; and
- both consume the same total-rate budget.

Rank nesting says nothing about containment or disjointness with the composite.
A valid nesting test must start from the literal decoded-composite residual on
the same sources, fit/apply IDRE there, serialize one combined object, and redo
one joint rate/read allocation.  This package makes no additive, containment,
or composite-pass claim.

## Source-only reproduction

The default path reads only the already authenticated aggregate JSON and its
receipt.  It does not import CuPy and cannot accept payload arguments:

```bash
/usr/bin/python3.12 -B -I stage0_gate.py \
  --prior-result ../shared_subspace_gate/result.json \
  --prior-verification ../shared_subspace_gate/verification_receipt.json \
  --output /tmp/kbvq_idre_prior_replay.json
```

The standard-library verifier does not import the gate or CuPy:

```bash
/usr/bin/python3.12 -B -I verify_design.py --package .
/usr/bin/python3.12 -B -I test_source_only.py -v
```

Only after explicit root coordination, the independent RTX/CuPy replay is:

```bash
/workspace/int2-cupy-venv/bin/python -B -I stage0_gate.py \
  --prior-result ../shared_subspace_gate/result.json \
  --prior-verification ../shared_subspace_gate/verification_receipt.json \
  --authorize-independent-payload-replay \
  --authorization-token ROOT_COORDINATED_REPLAY_V0 \
  --manifest /workspace/INT2__compression/agent_rd_structure_diag_cross_expert_sources.json \
  --root /workspace/INT2__compression \
  --output /workspace/kbvq_idre_raw_mse_replay_v0.json
```

At seal time that coordinated command had **not** been run.

## Claim boundary

This package rejects the physically charged FP16 IDRE cells under the declared
two-role residual Gaussian model on held-out layer-15 auxiliary Qwen weights.
It does not reject activation-weighted IDRE, BCOS, model-quality gains,
within-residual-mode anisotropy before the coordinated replay, nonlinear shared
decoders, or a valid joint composition evaluated on the composite residual.
