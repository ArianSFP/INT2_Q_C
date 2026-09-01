# High-dimensional additive VQ early-kill screen

This is a CPU-only source experiment, not a serialized-codec claim. Codebooks are
trained on five experts and evaluated on the held-out sixth; the same pipeline is
independently fitted to moment-matched iid Gaussian controls. All tables and compact
per-matrix scalars are charged, while the favorable decision bound additionally grants
two fold standard errors and a fixed numerical allowance.

Required advantage: `s >= 0.160964047443681` bpw, equivalently `F=2^(-2s) <= 0.8`.

## Reproduction

Run from `/workspace/INT2__compression` on the supplied RunPod. The program is
NumPy/CPU-only and the bounded nine-cell run consumed 71.74 aggregate experiment
seconds:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python \
  additive_vq_oracle/additive_vq_screen.py \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --output additive_vq_oracle/final_xklt \
  --representation xklt --dimensions 8,16,32 --alphabets 2,3,4 \
  --sample-vectors-per-matrix 2048 --nominal-rate 2.25 \
  --lloyd-iterations 3 --refit-rounds 1 --sweeps 1 --batch 4096
```

The independent verifier recomputes the plan seal, all 18 live source hashes,
the nine rate/cost ledgers, every `F=2^(-2s)` identity, fold coverage, and the
read-amplification gates:

```bash
python additive_vq_oracle/verify_additive_vq_screen.py \
  --result additive_vq_oracle/final_xklt/additive_vq_screen_result.json \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --receipt additive_vq_oracle/final_xklt/verification_receipt.json
```

| architecture | R physical | source D | matched s | optimistic s | optimistic F | exact F vs Shannon | cold read |
|---|---:|---:|---:|---:|---:|---:|---:|
| role-conditioned-additive-rvq-d8-k3-m11 | 2.179841 | 0.12699057 | -0.015589 | 0.005533 | 0.992359 | 2.607154 | 1.001285x |
| role-conditioned-additive-rvq-d8-k4-m9 | 2.250558 | 0.11979258 | -0.021030 | -0.006600 | 1.009192 | 2.712696 | 1.001353x |
| role-conditioned-additive-rvq-d8-k2-m18 | 2.250558 | 0.11489658 | -0.031507 | -0.010531 | 1.014706 | 2.601826 | 1.001353x |
| role-conditioned-additive-rvq-d16-k2-m36 | 2.252023 | 0.11103331 | -0.023895 | -0.010790 | 1.015070 | 2.519454 | 1.005260x |
| role-conditioned-additive-rvq-d16-k4-m18 | 2.252023 | 0.11764833 | -0.023278 | -0.011712 | 1.016368 | 2.669555 | 1.005260x |
| role-conditioned-additive-rvq-d16-k3-m23 | 2.280326 | 0.11374243 | -0.027926 | -0.012034 | 1.016822 | 2.684203 | 1.004980x |
| role-conditioned-additive-rvq-d32-k4-m36 | 2.257883 | 0.11616172 | -0.029954 | -0.017676 | 1.024807 | 2.657320 | 1.020885x |
| role-conditioned-additive-rvq-d32-k3-m45 | 2.236248 | 0.11673174 | -0.031145 | -0.018134 | 1.025457 | 2.591459 | 1.019768x |
| role-conditioned-additive-rvq-d32-k2-m72 | 2.257883 | 0.10920263 | -0.031819 | -0.018983 | 1.026666 | 2.498123 | 1.020885x |

## Decision

The most favorable tested result reaches only `3.437%` of the required advantage after the deliberately generous allowance. Its identity is `F = 2^(-2*0.005532936884) = 0.992359062322`. The branch is therefore rejected early; increasing training effort cannot plausibly close
the remaining order-of-magnitude gap, and the uncalibrated absolute finite-dimensional
quantizers are farther from the Gaussian Shannon curve still.

## Claim boundary

This rejects the tested adjacent-coordinate, role-conditioned additive residual VQ family
at dimensions 8/16/32 and binary/ternary/quaternary alphabets. It is not a converse for
arbitrary semantic reordering, extremely large unstructured codebooks, or joint coding
across experts. The oracle per-matrix reconstruction gain is stored and charged; it makes
the screen more favorable to the candidate.

Full source hashes, plan/header seals, fold results, costs, and exact identities are in
`additive_vq_screen_result.json`.
