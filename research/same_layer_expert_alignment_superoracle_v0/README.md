# Same-layer expert-alignment super-oracle v0

This source-only CuPy gate tests a structural opportunity not covered by the
cross-layer neuron-permutation audit or unaligned shared-matrix PCA: hidden
neurons copied during MoE upcycling may survive under different per-expert
orders.

The gate is intentionally more permissive than any codec.  For each Up and
Down row of each of sixteen authenticated layer-15 experts, it may choose any
of the 11,520 rows in all fifteen other experts, independently by role and
with reuse.  It then fits an exact FP64 slope and intercept.  References,
mappings, coefficients, and reads are all free.  Float32 CuPy finds the best
absolute correlation and the selected regression is replayed in FP64; a very
generous additional 0.001 absolute capture is credited before the decision.

The existing best ideal composite is short by `0.11356063457 bpw`, equivalent
to a required residual-energy capture of `0.14566207552117194` if this were
the sole missing module.  Missing that threshold under the illegal oracle
kills finite same-layer Up/Down reference engineering early.  It does not
cover Gate ancestry, nonlinear generative sharing, or activation-weighted
functional compression.

Verify the inert package before any payload access:

```bash
/usr/bin/python3 -B -I verify_source.py \
  --package . --workspace /workspace/INT2__compression
```

Coordinated RunPod command:

```bash
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 \
  /workspace/int2-cupy-venv/bin/python -B -I same_layer_alignment_oracle.py \
  --workspace /workspace/INT2__compression \
  --output /workspace/same_layer_expert_alignment_superoracle_v0_result.json
```
