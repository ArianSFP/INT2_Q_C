# Independent CBIB-1 source audit

This is a hostile, payload-blind review of producer manifest
`1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a`
and source root
`18a4043e99b17cfa535f4a6c2930f2c1ac42eff092f4e5d61b9408b1986f457e`.

Verdict: `PASS_INDEPENDENT_SOURCE_ELIGIBLE_FOR_SEPARATE_DEPLOYMENT_REVIEW`.

The audit independently reproduced closure and all 14 producer tests, then
used separate reference calculations to check all-coordinate cross-fitting,
hard-EM count sufficiency, the explicit latent/private factorization, KT NLL,
equal-partition cardinality, selector/model/frame/scale charges, source-first
control order, affine-control marginal preservation, page capacity, and both
physical and non-padding read denominators. A synthetic non-modal latent was
recovered in every fold. Its small fixture has excellent ideal structure but a
negative fully charged result, correctly demonstrating that headers are not
silently waived.

A fresh independent RTX 3060 process reproduced exact CPU/CuPy counts and
assignments; the maximum codelength difference was
`9.094947017729282e-13` bits. The producer's separate RTX receipt was inspected
and hash-pinned but was not treated as independent authority. A source-free
RunPod replay was attempted; the sealed package was absent remotely and the
execution sandbox denied source export. It was therefore not executed and no
workaround was attempted. This is not a source defect.

No Qwen/model file was opened. This package is not deployment authority, a
Qwen result, a finite entropy codec, or evidence for the target `F <= 0.8`.
The source package remains on literal HOLD and its read result is explicitly an
ideal capacity envelope.

## Deployment preconditions

Before any payload run, a separate manifest-pinned deployment must:

1. pin this exact producer manifest/root and receive an independent deployment
   review; the source package itself must not be edited or enabled;
2. bind the exact input panel, quantizer/label mapping, shapes, scale stream,
   and output path before access, with output required absent;
3. derive and audit `scale_bytes_per_expert` from that bound input/container;
   it must not be a convenient caller-selected value;
4. preserve the source-first order and require the exact success status plus
   `capacity_ok=true` and `strictly_below_2x=true` at an exact endpoint;
5. call the page ledger only on the flat fold partitions emitted by
   `crossfit_group_size`/`packet_requirements`, not on arbitrary memberships;
6. state that the frozen maximum matched-control correction is scoped per
   predeclared group-size family. A bank-wide correction would be a new source
   design and requires resealing and review;
7. perform source-free CPU/CuPy parity in the deployment environment, then run
   the fixed controls only after source and read survival; and
8. continue to label all results ideal until a literal independently decoded
   packet charges arithmetic termination, framing, pages, and coder loss.

Run the independent evidence test with the exact external producer pin:

```powershell
$audit = "research/same_layer_clustered_ib_entropy_gate_v0_independent_source_audit_20260903"
$source = "research/same_layer_clustered_ib_entropy_gate_v0"
$gpu = "research/same_layer_clustered_ib_entropy_gate_v0_local_rtx3060_preflight_20260903/RUN_RECEIPT.json"
python -B "$audit/audit_source.py" --source-package $source `
  --manifest-sha256 1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a `
  --producer-gpu-receipt $gpu
```
