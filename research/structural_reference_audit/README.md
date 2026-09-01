# Structural-reference auxiliary package audit

This directory contains a source-free independent audit of:

- `research/dense_upcycle_reference`
- `research/permutation_aligned_expert_template`

No model payload was opened, no experiment file was changed, and no CuPy/CUDA
context was imported or initialized. The audit used only copied code, JSON
results/manifests, READMEs, and existing provenance evidence.

## Verdicts

| Package | Integrity and arithmetic | Scientific verdict |
|---|---|---|
| Dense upcycle reference | Pass | Scoped early kill is justified for the two-layer affine neuron-ancestor screen |
| Permutation-aligned template | Pass | Strong negative evidence for the exact one-pair joint-scalar cell, but the claimed favorable upper bound and broader family kill are blocked |

Both packages use authenticated auxiliary layer-15 identities outside the six
pinned layer/expert pairs. Their recorded source hashes match the pre-existing
auxiliary result. The result seals, energy/capture identities, control
subtractions, and threshold arithmetic all replay exactly.

## Dense upcycle result

The rectangular Hungarian score is the sum of explained SSE for independent
centered Up/Down affine regressions, so it matches the implemented distortion
objective. The opportunity gap is large even before relying on the control:

```text
required incremental capture  0.14566207553404276
best raw capture              0.005129678392775006   (28.396x short)
max scramble control          0.005003297765905912
control-corrected capture     0.000126380626869094   (1152.57x short)
```

This supports stopping the tested public Qwen3-1.7B layers 9/15 affine
neuron-ancestor path. It is not a converse for unsearched layers, nonlinear or
multi-neuron references, private ancestors, or unrelated training lineages.

Numerical wording should remain modest: the assignment dots are FP32 CuPy GEMM
outputs before FP64 division, and the centered reconstruction uses an
FP32-rounded mean while the hashed beta uses the FP64 mean. Thus the screen is
not literally an exact-FP64 fit. Independent signed scrambles are also applied
to Up and Down, so the control does not preserve cross-role target structure.
Neither caveat is needed to obtain the 28.4x raw opportunity gap.

## Permutation result blocker

The code assigns neurons by maximizing unweighted squared cosine. After the
through-origin least-squares fit, however, explained raw SSE is

```text
(reference_i dot target_j)^2 / ||reference_i||^2
    = ||target_j||^2 * cosine_squared(reference_i, target_j).
```

The omitted target-energy factor can change the Hungarian assignment. For
example, with target energies `[100, 1]` and cosine-squared matrix
`[[0.6, 0.9], [0.5, 0.0]]`, unweighted cosine selects the swap (`1.4 > 0.6`),
while explained SSE selects identity (`60 > 50.9`). The result stores neither
the full score/energy matrices nor a bound on the correctly weighted optimum.

The cell is also narrower than the claimed shared-template upper screen: it
fits one through-origin scalar jointly over concatenated Up/Down, with no
separate role scalars, intercepts, or Gate role. These legal variants strictly
dominate it. The two Gaussian controls match global raw energy before an FP32
cast, but do not match source means, row-energy distributions, cross-role
structure, or a calibrated null quantile.

The stored numbers are still useful prioritization evidence:

```text
required incremental capture  0.14566207553404276
raw capture                   0.002965283193652746   (49.122x short)
max Gaussian control          0.002590185248450450
control-corrected capture     0.000375097945202296   (388.331x short)
```

Accordingly, pausing the exact `L15/E0 -> L15/E8`, Up+Down, unweighted-cosine,
one-joint-scalar cell is rational. This package alone does not certify a kill
of permutation-aligned expert templates. A prior committed oracle supplies
stronger corroborating evidence by testing Gate/Up/Down, all 30 directed pinned
pairs, an energy-objective role-wise assignment, and an illegal many-to-one
ceiling; that evidence does not retroactively fix this package.

## Required source-language downgrade

The experiment files were deliberately not edited by this audit. Before
checkpointing them as authoritative, update these claims and recompute the
result seal/hashes:

1. In `research/permutation_aligned_expert_template/README.md`, replace
   `Decision: early kill` with `Decision: heuristic pause of the tested
   unweighted-cosine joint-scalar cell; not a family kill`.
2. Replace `globally optimal 768-by-768 assignment` / `optimal 768-neuron
   permutation` with `Hungarian assignment maximizing unweighted cosine^2;
   not guaranteed to maximize raw-MSE capture`.
3. Replace the conclusion that a cross-fitted multi-expert codec is not
   justified with the narrower statement that this exact directional cell is
   deprioritized; broader role-wise/Gate variants require the existing stronger
   oracle or a corrected v2 screen.
4. In `pair_result.json`, rename the decision to something such as
   `HEURISTIC_PAUSE_UNWEIGHTED_COSINE_JOINT_SCALAR_CELL`, update
   `claim_boundary`, and remove `optimal 768-neuron permutation` from
   `favorable_grants`. Recompute its canonical seal and file hash.
5. In `permutation_pair_screen.py`, either describe the current objective
   exactly or make a v2 that assigns with `dot^2 / reference_energy`, fits free
   role-wise coefficients, and includes Gate before calling it a favorable
   expert-template upper screen.

## Sealed receipts

Dense receipt:

- File SHA-256:
  `b3015194876a7504e90334e2eade9ab52db2bfd891e680e8b625a867fbda9fb5`
- Canonical unsigned seal:
  `c654e8aebf6d6327a9420eb6266ad874232c0c56eb58d9a5db9aaac74cb67c3e`

Permutation receipt:

- File SHA-256:
  `d375777d76f4b89e92164ea2e36fc0bcb47fe66dacd5e68d68599fcdf0526ed2`
- Canonical unsigned seal:
  `ef4319f2be11050eb379d4ed8d90d1f8a771326f0c257ec6873a5766242ed93a`

Verify copied package artifacts and arithmetic with:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  python3 -B verify_packages.py --repo /path/to/INT2_Q_C
```

Verify the receipt seals and exact bytes with:

```sh
python3 -B verify_receipts.py \
  dense_upcycle_reference_audit.json \
  permutation_aligned_expert_template_audit.json
sha256sum -c dense_upcycle_reference_audit.sha256
sha256sum -c permutation_aligned_expert_template_audit.sha256
```
