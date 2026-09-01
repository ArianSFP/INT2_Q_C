# Router-visible side-information oracle

## Verdict

**Hard kill.**  The same-layer Qwen router is available before an MoE expert
is fetched, so it is unusually favorable decoder-visible side information.
Nevertheless, the strongest tested router-derived subspace explains only
`7.6683326836%` of source energy with `6.25%` of the dimensions.  Even after
granting free per-matrix mode selection, exact orthonormal arithmetic, and
infinite-block Gaussian reverse waterfilling, the best result is

```text
F = MSE * 2^(2R) = 0.9984917873401222
s = -0.5 log2(F) = 0.0010887667149184358 bpw
```

The goal requires `F <= 0.8`, or `s >= 0.16096404744368115 bpw`.  This branch
therefore supplies only `0.6764%` of the required rate-equivalent advantage
under assumptions more favorable than an implementable codec.  It must not
consume a production GPU encode.

At `R=2.5`, the oracle MSE is `0.03120286835437882`; the required MSE is
`0.025`.

## Bound inputs

The result binds:

- the sealed 18-matrix Qwen3-30B-A3B source plan, lock
  `99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`;
- all eighteen original BF16 source hashes;
- the six exact `128 x 2048` same-layer router matrices and their HTTP shard
  byte ranges; and
- every projection result for eight ranks and three legal router-only basis
  families.

The tested ranks are `k={1,2,4,8,16,32,64,128}`.  The basis families are:

1. the right singular vectors of the full layer router;
2. a DCT basis ordered by the routed expert's router row; and
3. router-PCA directions multiplicatively modulated by that router row.

For each target matrix, the oracle is also allowed to choose the best of these
three modes for free.  This adaptive choice leaks source information and is
intentionally optimistic.  A valid format would have to encode it.

## Calculation

For a rank-`k` basis, let `d=k/2048` and let `e` be the pooled fraction of
source energy captured by the exact projection.  The probe treats the basis
and orthogonal residual as two independent Gaussian components with dimension
fractions `(d,1-d)` and energy fractions `(e,1-e)`.  It then solves the exact
two-component reverse-waterfilling allocation at `R={2.15,2.30,2.50}`.

This calculation grants away all basis bytes, labels, mode bits, scales,
framing, finite-vector loss, and finite-code loss.  Consequently, failure is a
sound early-kill for these basis families; success would only have promoted a
candidate for a real codec and would not itself have established one.

## Reproduction

The sealed CPU-only RunPod command was:

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/router_sideinfo_oracle/router_sideinfo_oracle.py \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --router-dir /workspace/INT2__compression/qwen_aux_context_tensors/router_blocks \
  --output /workspace/INT2__compression/router_sideinfo_oracle/router_sideinfo_result.json
```

Verify the plan seal, all available source/router bytes, and independently
recompute every waterfill/F/s identity with:

```bash
/workspace/int2-cupy-venv/bin/python verify_result.py \
  --result router_sideinfo_result.json \
  --plan-dir /workspace/INT2__compression/strata_expert_affine_milestone_v1 \
  --router-dir /workspace/INT2__compression/qwen_aux_context_tensors/router_blocks \
  --receipt verification_receipt.json
```

The independent verifier passed `18/18` source files, `6/6` router files,
`32/32` curves, and `96/96` rate cells.  Frozen evidence hashes are listed in
`ARTIFACT_HASHES.json`; in particular, the result SHA-256 is
`846dc75f43eb796d8427e8426309baf50558a194ccce469a919cae499c67f90a`.

## Claim boundary

This result rejects the three listed linear/nonlinear router-derived subspace
families under a stronger-than-real codec oracle.  It does not reject an
arbitrary conditional generative model of expert weights, nor does it claim a
universal converse.  It is source-domain relative MSE evidence, not a
perplexity or activation-weighted result.
