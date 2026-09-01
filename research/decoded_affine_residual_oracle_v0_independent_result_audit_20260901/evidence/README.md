# Decoded affine residual correction envelope

This bounded CuPy screen asks whether the independently decoded 2.5-bpw
STRATA expert-affine reconstruction has enough systematic scale or bias error
for a tiny BCOS-like correction stream to close the strict `F <= 0.8` target.

It tests exact source-fitted scale-only, bias-only, and affine corrections on
contiguous widths 2,048, 512, 128, and 32, plus a row-and-column additive-bias
model.  Every coefficient is unrealistically kept in FP64 while only its
nominal FP16 storage is charged.  It also grants that the measured correction
fraction transfers unchanged after reducing the coarse payload by that side
rate.  Failure is therefore an architecture-scoped hard kill; success would
only authorize a finite quantized-coefficient experiment.

Run on the provided RTX 5090 after checking that no coordinated job is active:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B \
  decoded_affine_oracle.py \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --output /workspace/decoded_affine_residual_oracle_v0_result.json
```

This screen is source-domain only and does not claim activation-aware model
quality, a finite codec, or a universal result for nonlinear corrections.
