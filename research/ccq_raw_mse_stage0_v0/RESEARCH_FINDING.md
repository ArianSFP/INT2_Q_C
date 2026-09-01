# CCQ primary-source and containment finding

## Finding

CCQ is sufficiently distinct to justify one narrow raw-weight-MSE gate, but
only for the Code-Cluster cell. It is not evidence for the strict target.

The primary paper defines a configuration `(L,N,S)`. A length-`N` integer
vector is represented by `T=L+(N-1)S` bits; adjacent `L`-bit states overlap by
`L-S` bits. CCQ explicitly changes convolutional/trellis coding into a finite
vector-codebook search. Decoding does not load that table: shifts and masks
recover the overlapping states from the stored integer.

At the headline low rate, `(L=6,N=4,S=3)` first produces a 15-bit structured
code. Code Cluster then uniformly requantizes the code values separately for
each output channel to a byte. The released decoder computes

```text
c15 = round_fmaf(byte, code_scale_float32, code_zp_float32)
z = [((c15 >> 9) & 63), ((c15 >> 6) & 63),
     ((c15 >> 3) & 63), (c15 & 63)] - 32
w_hat = z * uint4_local_scale * fp16_super_scale
```

The paper is weight-only PTQ and searches codewords using MSE, but its reported
end results are downstream benchmark scores and memory/operator measurements.
It does not publish pooled raw-source relative MSE, `F=MSE*2^(2R)`, a matched
Gaussian control, or evidence for this repository's 20%-below-Gaussian target.

Primary sources:

- paper: <https://arxiv.org/abs/2507.07145v1>
- official FastDeploy repository: <https://github.com/PaddlePaddle/FastDeploy>
- pinned released loader: <https://github.com/PaddlePaddle/FastDeploy/blob/f5562df9fbd543a63dc28bb8e5709cb6d90e1707/fastdeploy/model_executor/layers/moe/fused_moe_wint2_backend.py>
- pinned released dequantizer: <https://github.com/PaddlePaddle/FastDeploy/blob/f5562df9fbd543a63dc28bb8e5709cb6d90e1707/custom_ops/gpu_ops/cutlass_extensions/gemm/threadblock/wint2x_unzip.h>

The reviewed official commit exposes the prequantized tensor shapes and
runtime decode, not the offline CCQ encoder. Therefore this package freezes a
paper-derived min/max Code-Cluster encoder. It is not called an official-code
reproduction.

## Exact side-rate correction

The paper's `2.06 bpw` is the asymptotic `2 + 4/64 = 2.0625 bpw` byte and
local-scale ledger. The released implementation also requires, per output
channel, two FP32 values (`code_scale`, `code_zp`) and one BF16/FP16
`super_scale`: 10 bytes.

In released `K x N` orientation, Gate and Up are `2048 x 768`; Down is
`768 x 2048`. For one three-matrix expert:

| Field | Exact bytes |
|---|---:|
| Three uint8 code matrices (four weights/byte) | 1,179,648 |
| Packed uint4 local scales (one/64 weights) | 36,864 |
| Two FP32 code fields over 3,584 output channels | 28,672 |
| FP16 super-scales over 3,584 output channels | 7,168 |
| Expert header | 64 |
| **Expert total** | **1,252,416** |

With one 4,096-byte global header, six experts require 7,518,592 fixed bytes,
or exactly `2.1245298032407407 bpw`. The first stage must therefore reach
`q <= 0.0420722358191473` even when every remaining bit is granted to an ideal
Gaussian residual.

The paper's 2.5-bpw hybrid cell is already outside this task after exact side
charging. Its 64-weight/160-bit payload is 1,474,560 bytes per expert, and the
released decoder still needs 7,168 FP16 channel super-scale bytes. Six-expert
framing makes the prefix `2.5134186921296297 bpw > 2.5`. No payload experiment
is justified for that cell.

## Distinction from the local ledger

- **POLARIS/STRATA:** those are long sequential RHT/polar-lattice/arithmetic
  streams. CCQ independently searches four-weight structured vectors and uses
  no RHT, polar decisions, or causal stream.
- **Additive VQ:** CCQ has one overlapping-window structured word, not a sum of
  independently indexed residual codewords.
- **Bitplane context:** CCQ uses fixed-width bytes and arithmetic decode, not
  causal entropy prediction of neighboring labels.
- **QSB:** CCQ is deterministic nearest structured coding; it has no stochastic
  channel, shared-randomness correction, or multiplicative binary decoder.
- **Direct-output VQ:** materializing one channel's decoder gives a constrained
  256-by-4 table. That observation does not contain CCQ in the tested
  K=32768,d=8 direct gate: CCQ's table varies by expert/output channel from two
  floats plus local scales, while that gate stores one table shared across all
  roles and experts. An arbitrary per-channel table would contain CCQ
  representationally, but its physical side cost was not tested and is much
  larger.

This is a genuine but narrow distinction: CCQ is principally a structured,
lookup-free, channel-conditioned VQ cell, not a new model-wide learned source
model.
