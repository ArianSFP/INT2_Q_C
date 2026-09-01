# QSB-PTQ-v0 source-only stage-0 design

Verdict: **READY FOR COORDINATED STAGE-0; NO GPU OR NUMERIC PANEL PAYLOAD WAS USED TO PREPARE OR SEAL THIS PACKAGE.**

This package freezes a deliberately optimistic screen for a stochastic-binary
post-training codec on the authenticated 18-matrix Qwen panel. It was inspired
by the binary-channel-simulation viewpoint in *SoftBinary Coding*, arXiv
2606.29578v1, but it is neither that paper's implementation nor a reproduction
of any reported result. The primary version-pinned reference is
<https://arxiv.org/abs/2606.29578v1>.

## Frozen construction

`QSB-PTQ-v0` divides every 768-by-2048 scored matrix into 64-value blocks. A
fixed signed projection produces 160 Bernoulli probabilities per block. A
shared-randomness draw produces 160 bipolar bits, and 96 frozen pairwise
products augment them to 256 decoder features. The eventual decoder is one
shared 64-by-256 Q7 affine map. The exact experiment score uses its binary64
affine reconstruction before any deployment cast.

This is genuinely different from additive VQ: there is no codebook or nearest
codeword, no residual-stage sum, and the decoder is nonlinear in the stochastic
bits because of the 96 multiplicative interactions. Its reconstruction set is
implicit rather than a stored vector dictionary.

The future operational encoder is a separately auditable proposal: 45 exact
segments of 2^18 binary channels per expert, seeded permutation/polar
transforms, shared randomness, an XOR correction, and an integer range coder.
This stage does not implement that encoder. It grants ideal empirical KL with
zero finite-block, table, framing, and entropy-coder overhead. Consequently its
rate number is an optimistic oracle, never physical evidence.

## Exact physical ledgers

All future messages must fit a fixed reservoir. Overflow rejects the cell;
unused bytes are zero-filled and still charged. Every expert has one 4096-byte
header. The shared packet is exactly 24,576 bytes: 4096 schema/hash/CRC bytes,
16,384 Q7 decoder bytes, 2048 bias/scale/prior bytes, 1024 transform/PRG bytes,
and 1024 probability-table/padding bytes.

| Cell | Expert pages | Payload bytes/expert | Container bytes | Exact physical bpw | Cold page reads |
|---|---:|---:|---:|---:|---:|
| QSB215 | 309 | 1,261,568 | 7,618,560 | 155/72 = 2.1527777778 | 63/62 = 1.0161290323 |
| QSB230 | 331 | 1,351,680 | 8,159,232 | 83/36 = 2.3055555556 | 337/332 = 1.0150602410 |
| QSB250 | 359 | 1,466,368 | 8,847,360 | 5/2 = 2.5 | 73/72 = 1.0138888889 |

The metadata contribution is exactly 1/72 bpw for every cell. Physical rates
are inside [2.15, 2.5], and all frozen cold-read ratios are below 2x. The exact
decision objective is `F = relative_mse * 2^(2R) <= 0.8`, with every byte and
all padding charged.

## Favorable source-leaking stage-0 bound

The runner first binds and hashes the exact plan and all 18 BF16 sources. It
selects alpha and the Bernoulli prior only on experts 0, 2, and 4. It then fits
an arbitrary FP64 decoder separately on every scored matrix and grants the best
of all three frozen common-randomness seeds. This knowingly fits the score
source and includes the eventual one-seed shared Q7 affine decoder family, so it
is a favorable topology upper bound, not held-out compression performance.

Strict ordering is frozen:

1. Kill a rate cell before decoder fitting if any expert's ideal KL exceeds 97%
   of its fixed payload reservoir.
2. After the complete favorable Qwen oracle, kill before controls if aggregate
   capture plus three delete-expert jackknife SE is below the exact cell target.
   No raw-SSE shortcut may bypass this uncertainty rule.
3. Only survivors run eight independent FP64 Gaussian controls, each matched to
   every source matrix's empirical mean and centered energy and given identical
   alpha fitting, three seeds, and source-fit decoder effort.
4. Hold a cell for a future operational implementation only if Qwen's
   lower-three-SE capture exceeds the strongest control upper-three-SE and each
   expert and role capture exceeds the strongest corresponding control fold.

A hold still is not compression evidence and creates no authorization to access
fresh validation data.

## Gaussian, TCQ, and Shannon claims are different

The matched Gaussian runs are empirical stage-0 algorithm controls. They emit
no codec bytes and are not a lower bound. An operational Gaussian comparison
must run the eventual byte-emitting QSB codec. An operational TCQ comparison
must independently implement a finite-state TCQ, emit and decode its actual
bytes, and charge its model, framing, and padding.

The Shannon Gaussian MSE expression `D_G(R) = sigma^2 * 2^(-2R)` is an
asymptotic information-theoretic lower bound for an ideal memoryless Gaussian
source. It emits no bytes and is neither TCQ nor an empirical Gaussian control.
No paper curve or Gaussian outcome may be described as Qwen evidence, and no
finite-sample stage-0 oracle can establish a Shannon-limit claim.

## Source-only verification

From this directory, using a canonical standard Python interpreter:

```text
python verify_design.py --package .
python -m unittest -v test_source_only.py
```

The verifier requires the exact eight-file, regular-file, non-link closure;
validates the manifest and canonical receipt; recomputes topology, all three
physical-rate/cold-read ledgers, objective thresholds, splits, source bindings,
gates, and claim boundaries; and statically compiles the runner without
importing CuPy or opening numeric payloads.

## Future coordinated stage-0 command

Do not run this until the coordinator supplies and authorizes exact isolated
paths. The command has no source defaults and refuses an existing output:

```text
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 stage0_screen.py \
  --plan /ABSOLUTE/HELD/plan.lock.json \
  --source-root /ABSOLUTE/HELD/panel-root \
  --output /ABSOLUTE/ABSENT/qsb_stage0_result.json
```

The output is a result ledger only, not compressed weights. It records all
source hashes, software/runtime versions, the exact design and runner hashes,
rate/oracle/control decisions, zero fresh-validation access, and zero compressed
artifacts.
