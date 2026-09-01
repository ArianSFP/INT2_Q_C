# Cross-fitted nonparametric Qwen rate--distortion oracle

This directory contains a CPU-only early-kill experiment for a specific open
question left by the existing Qwen quantization work: can scalar marginal or
short adjacent-vector non-Gaussianity supply the `0.1609640474` bit/weight
advantage required for MSE 20% below the iid-Gaussian rate--distortion line?

It is intentionally different from the existing GGD, histogram-KL, kurtosis,
shape-class, and covariance screens. It fits a nonparametric stochastic test
channel with the Blahut--Arimoto update and evaluates it on complete held-out
expert triplets. Raw Gate/Up/Down and the already-sealed Gate/XKLT0/XKLT1
coordinates are tested with scalar, adjacent-2D, and adjacent-4D alphabets.

The strict cross-fit rate is

```text
E_test KL(P_beta(reconstruction | x) || q_train(reconstruction)).
```

The output prior is therefore not silently refit on the held-out matrix. A
moment-matched Gaussian control goes through exactly the same finite table and
solver. The most favorable score divides out *all* Gaussian-control loss,
ignores the side table first, then reports a charged version with explicit
normalization, rotation, reconstruction-table, and frequency-table bits.

Run on the canonical RunPod without using the GPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python rd_nonparametric_oracle/nonparametric_ba_oracle.py \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --output rd_nonparametric_oracle/qwen_nonparametric_ba_result.json
```

A small dependency and numerical smoke test is available locally:

```bash
python rd_nonparametric_oracle/nonparametric_ba_oracle.py --self-test
```

The experiment is a bounded empirical architecture gate, not a universal
information-theoretic converse. A negative result rejects stationary scalar
and adjacent 2-D/4-D test channels under these normalizations; it does not
reject long-range deterministic or semantic structure.
