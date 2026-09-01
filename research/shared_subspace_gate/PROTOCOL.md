# Shared-subspace raw-MSE gate (frozen before payload access)

## Question

Test the raw source-MSE analogue of KBVQ-MoE's cross-expert shared SVD:
does a right, left, or two-sided subspace learned from whole auxiliary Qwen
experts concentrate enough energy in untouched experts to reach

```text
F = D * 2^(2R) <= 0.8,  2.15 <= R <= 2.5,
```

after charging a decoder-visible FP16-sized basis and requiring cold expert
reads below 2x?

This is an auxiliary early gate, not a claim about KBVQ-MoE's activation-
weighted objective and not a final codec result.

## Immutable data split

- Model: `Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Source manifest SHA-256:
  `4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782`.
- Layer: 15; roles: canonical `up` and transposed `down`.
- Basis-fit experts: `0,8,16,32,40,48,64,72,80,96,104,112`.
- Untouched validation experts: `24,56,88,120`.
- The pinned 18-matrix evaluation panel is forbidden.

## Frozen candidates

Fit FP32 covariances on fit experts only. Test joint-role and role-specific:

- shared right bases, ranks `8,16,32,64,96,128,192,256`;
- shared left bases, ranks `4,8,16,32,48,64,96,128`;
- Cartesian two-sided combinations of those rank grids.

The favorable oracle treats basis vectors as exact after charging their
FP16 literal size. It orthogonally decomposes each untouched matrix into one,
two, or four components and performs one ideal Gaussian reverse-waterfill.
Because exact vectors and asymptotic coding favor the hypothesis, failure is
an early kill. A survivor must next be replayed with the serialized FP16 basis
and a finite residual codec.

## Physical and read ledger

- Basis bytes are amortized across all 128 experts of this layer for physical
  bpw, but the entire required basis is counted in a cold single-expert read.
- Add 512 framing bits per expert.
- Coefficient payload is the requested physical rate minus amortized side.
- Report only candidates with positive payload and cold read amplification
  below 2x.
- No activation, router, target, or validation-derived basis is permitted.

## Decision

- Promote if an untouched-expert row has `F <= 0.8` and reads `<2x`.
- Retain as a composite lead if `F <= 0.90`.
- Otherwise kill this shared linear-subspace family before a production
  encode. This does not reject activation-weighted KBVQ-MoE or nonlinear
  shared decoders.

