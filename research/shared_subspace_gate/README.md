# Decoder-visible shared-subspace gate

## Result

**Hard kill.** A raw-MSE implementation of the cross-expert shared-subspace
idea does not explain the reported functional gains of activation-guided MoE
factorization on this objective.

The gate fits bases on twelve layer-15 auxiliary Qwen experts and evaluates
four untouched whole experts. The pinned 18-matrix panel is never opened. It
tests joint-role and role-specific left, right, and two-sided bases over 160
predeclared rank cells. Every cell receives exact basis arithmetic after only
an FP16-sized charge and an ideal continuous Gaussian reverse-waterfill.

| Quantity | Best legal row | Required |
|---|---:|---:|
| Geometry | role-specific right rank 16 | — |
| Physical rate | 2.5 bpw | 2.15–2.5 bpw |
| `F=MSE*2^(2R)` | **0.9981959259638405** | **<=0.8** |
| `s=-0.5log2(F)` | **0.001302539625239886 bpw** | **>=0.160964047443681 bpw** |
| Ideal relative MSE | 0.031193622686370017 | <=0.025 at 2.5 bpw |
| Cold expert read | 1.1322916666666667x | <2x |

At rank 16, the learned input basis captures 1.3859% of untouched Up energy
and 2.6670% of untouched Down energy while occupying 0.78125% of each
matrix's coordinates. That anisotropy is real but far too small after the
decoder-visible basis charge. All larger and two-sided cells are worse.

This does not contradict activation-guided methods such as
[KBVQ-MoE](https://arxiv.org/abs/2602.11184): those methods optimize expert
outputs under a calibration covariance and report model quality. This gate
asks the different, stricter question of unweighted source-coordinate MSE.

## Physical/read accounting

An FP16 basis is amortized across all 128 experts of the layer in physical
bpw. For a cold routed-expert fetch, however, the complete required basis is
read once along with that expert's coefficient stream. A 512-bit expert frame
is also charged. No resident-cache assumption is used. The best row remains
comfortably below 2x; missing source structure, not bandwidth, kills it.

## Reproduction

On the supplied RunPod with CuPy 14.2:

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/shared_subspace_gate/shared_subspace_gate.py \
  --root /workspace/INT2__compression \
  --manifest /workspace/INT2__compression/agent_rd_structure_diag_cross_expert_sources.json \
  --output /workspace/INT2__compression/INT2_Q_C/research/shared_subspace_gate/result.json

/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/shared_subspace_gate/verify_result.py \
  --root /workspace/INT2__compression \
  --manifest /workspace/INT2__compression/agent_rd_structure_diag_cross_expert_sources.json \
  --result /workspace/INT2__compression/INT2_Q_C/research/shared_subspace_gate/result.json \
  --output /workspace/INT2__compression/INT2_Q_C/research/shared_subspace_gate/verification_receipt.json
```

The CuPy run completed in 1.33 seconds on the RTX 5090. The independent
verifier rehashes all 32 source payloads, checks the manifest and internal
result seal, reconstructs all 480 physical/read/waterfill rows, and reselects
the aggregate decision.

## Claim boundary

This is a favorable ideal-RD auxiliary early gate, not an emitted codec. It
rejects shared linear input/output subspaces for raw source MSE under the
frozen ranks and accounting. It does not reject activation-weighted error,
nonlinear shared decoders, or a procedural basis requiring zero stored bytes.

