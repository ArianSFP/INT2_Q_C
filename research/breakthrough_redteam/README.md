# Breakthrough red-team: learned priors, binary factors, and semantic dictionaries

## Decision

**No evidence-backed GPU survivor remains for the stated raw source-MSE goal.**
The only scalable logical loophole is still a compact shared learned prior,
tested on whole held-out experts against an identically trained
moment-matched Gaussian control. Published neural codecs do not supply the
needed raw-MSE evidence, and the two stricter structural screens added here
both fail by wide margins. A later primary-source lead, BiSCo-LLM, motivated a
bounded shallow implicit-codebook gate; that gate has now been executed and
independently replayed, with a slightly negative Qwen-versus-Gaussian
advantage. See [`BISCO_BSQ_ASSESSMENT.md`](BISCO_BSQ_ASSESSMENT.md) for the
pre-run assessment and [`../bisco_raw_mse_oracle/`](../bisco_raw_mse_oracle/)
for the sealed result.

The exact target is

```text
F = D * 2^(2R) <= 0.8,
s = -0.5 log2(F) >= 0.160964047443681 bpw,
2.15 <= R <= 2.5,
cold per-expert read amplification < 2x.
```

The existing 2.5-bpw expert-local checkpoint has `s=0.0080921265`, so a new
mechanism must add about `0.15287192093 bpw`.

## 1. Shared learned weight prior / decoder

This remains logically unclosed but is not evidence-backed.
[Neural Weight Compression (NWC)](https://arxiv.org/abs/2510.11234) is the
closest primary precedent: it learns nonlinear analysis/synthesis transforms
and an entropy model from weight chunks. Its own raw-MSE ablation reports that
adding the learned transform to entropy-constrained scalar quantization
*slightly worsens MSE*, while improving perplexity. Its 2.13-bpw result is a
perplexity comparison, and its Qwen3-30B-A3B table reports MMLU after
calibration-driven recovery rather than pooled source-relative MSE.

The published NWC synthesis network uses width 512 and four residual blocks.
Even the minimal interpretation of four 512x512 hidden maps exceeds the cold
headroom here: the current 10/9 coefficient read leaves only 1,310,720 bytes
for every shared object read cold with one expert. A width-256 or aggressively
quantized decoder can fit, but it must first demonstrate at least
`s=0.160964` gross on whole held-out Qwen experts. Existing local evidence is
far smaller: the strongest optimistic PRISM covariance result is only about
`s=0.0208`.

A legitimate remaining falsification experiment is a same-dimensional
16/32-weight codec, shared across all experts, trained with complete target
layer/expert exclusion. Serialize a decoder below about 1.2 MB, count its
actual entropy stream, and compare its held-out Qwen rate-distortion curve to
the same architecture trained on matched Gaussian patches. Do not promote it
unless every fold is positive and gross `s>=0.18` before implementation
losses. This is a test proposal, not a survivor claim.

### BiSCo-LLM / two-stage BSQ

[BiSCo-LLM](https://arxiv.org/abs/2607.08643) maps each chunk to a binary
latent, reconstructs it with a category-shared nonlinear decoder, and applies
a second codec to the residual.  It is not strictly upper-bounded by the local
additive-VQ screen: the latter sums many tiny explicit codebooks, whereas a
nonlinear decoder defines an implicit codebook with up to `2^b` outputs.  The
completed nonlinear-flow probe is also not an analysis/synthesis codec.

The paper nevertheless reports recovered perplexity/task accuracy rather than
codec-only pooled raw MSE, and does not give enough decoder dimensions or
component bytes to reproduce its claimed real rate.  Its spherical
normalization does not change the stored sign pattern, so any source-MSE gain
must come from held-out generalization of the shared decoder.  The exact
favorable oracle and physical/read ledgers for `d=16/32/64` are frozen in
[`BISCO_BSQ_ASSESSMENT.md`](BISCO_BSQ_ASSESSMENT.md).  A 2.25-bpw code plus
shallow FP16 role/stage decoders amortized over only 128 experts projects
external cold reads of `1.020x`, `1.080x`, and `1.315x`; the required matched
Qwen-vs-Gaussian advantages are respectively `0.16135`, `0.16240`, and
`0.16657 bpw`, before any finite-dimensional Gaussian gap.

The subsequent frozen `d=16, h=64, 18+18` CuPy gate stopped at update 512.
Independent state-backed FP64 replay measured
`D_Qwen=0.11020813758494276`, `D_Gaussian=0.1096131632504529`, and
`s_match=-0.0039048473221413946`; every whole-expert fold was negative. The
production ledger remained favorable at `2.2503823174 bpw` and `1.0204278591x`
cold reads, but the source-specific mechanism was absent. This hard-kills the
frozen shallow cell before pinned-panel access, not arbitrary nonlinear
codebooks.

## 2. NanoQuant-style overcomplete binary factorization

[NanoQuant](https://arxiv.org/abs/2602.06694) represents a matrix as

```text
W_hat = diag(s1) U V^T diag(s2),  U,V in {-1,+1},  s1,s2 in FP16.
```

This branch was genuinely not upper-bounded by the existing low-rank,
Stiefel/Gram, tensor, or adjacent additive-VQ probes. At the permitted rates,
its factor rank is greater than the smaller matrix dimension, so a continuous
low-rank relaxation is exact and therefore vacuous.

### Exact 768x2048 physical ledger

| Rank | U bytes | V bytes | FP16 scale bytes | Total bytes | BPW |
|---:|---:|---:|---:|---:|---:|
| 1,185 | 113,760 | 303,360 | 5,632 | 422,752 | 2.1502278646 |
| 1,380 | 132,480 | 353,280 | 5,632 | 491,392 | 2.4993489583 |

At rank 1,380, the exact 2.5-bpw cap is 491,520 bytes per matrix, leaving 128
bytes for framing. A three-matrix expert occupies 1,474,176 bytes, leaving 384
bytes. A fused `V^T x` then `U z` implementation reads each factor and scale
once, so coefficient read amplification can be approximately **1.0x**.

### Favorable matched-Gaussian cut-factor screen

`cut_factor_oracle.py` ran on two whole held-out expert triplets (six full
matrices) chosen only by a frozen hash rule. Each greedy binary outer product
receives an optimally refit **uncharged FP32 coefficient**, making this more
favorable than the physical NanoQuant ledger along that axis. Identical
restarts and alternating-sign optimization are used for Qwen and a Gaussian
matrix matched in shape, mean, and RMS.

| Rank | Ledger BPW | Qwen relative MSE | Gaussian-control MSE | Qwen-vs-Gaussian `s` | Absolute Qwen `F` |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.9453125 | 0.44309527 | 0.47461759 | **0.04957432** | 1.64298 |
| 1,185 | 2.1502279 | 0.15880163 | 0.16896102 | **0.04473238** | 3.12911 |
| 1,380 | 2.4993490 | 0.11783660 | 0.12532581 | **0.04444790** | 3.76737 |

The source advantage peaks near `0.0496 bpw` and then plateaus; at the exact
rank-1,380 ledger it is only 27.61% of the required `s`. Even granting a future
factor solver perfect Gaussian shaping while preserving the measured source
ratio would give `F=2^(-2*0.04444790)=0.94024`, only 5.98% below Gaussian, not
20%. The actual greedy absolute `F` is much worse. This is not a mathematical
converse for globally optimized binary factors, but it fires the frozen early
kill gate and does not justify a larger production run.

NanoQuant's paper evaluates perplexity and zero-shot accuracy and uses
activation-aware block/model reconstruction. It does not report the required
normalized raw source-MSE, so those functional claims cannot override this
gate.

## 3. BTC-LLM binary codebooks

[BTC-LLM](https://aclanthology.org/2026.acl-long.1066/) stores a binary
codebook of `c` length-`v` patterns plus indices, costing

```text
v*c + ceil(log2(c))*n*m/v bits
```

before binarizer coefficients. A single stage still reconstructs only a
binary weight field and cannot usefully spend 2.15--2.5 bpw on source MSE;
its published 0.7--1.11-bit gains are perplexity/accuracy claims. Stacking
residual stages becomes an additive binary code. The existing d=8/16/32
additive-VQ screen is more expressive locally because it grants real-valued
codewords, yet every matched source-vs-Gaussian gain was negative. The
bitplane/context audit found only `0.0000896 bpw` legal charged redundancy.
BTC's learned function-preserving transform is useful for activation error,
but it is not free side information under original-coordinate source-MSE.

### LiftQuant is already structurally screened

[LiftQuant](https://arxiv.org/abs/2606.04050) decodes a lifted binary vector as
`M q`, with `q in {-1,+1}^D` and code rate `D/d`.  This is a special case of
the completed role-conditioned additive-VQ family: `D` independent binary
codebooks with two arbitrary real `d`-vectors each can represent an affine
offset plus `M q`, and are strictly more expressive.  The rate-matched local
cells `d=16,D=36` and `d=32,D=72` both had negative Qwen-vs-Gaussian gains;
even their generous uncertainty bounds were negative (`s=-0.01079` and
`-0.01898`).  LiftQuant's paper reports perplexity/task behavior and designs
its projection for a Gaussian source.  It supplies no raw source-MSE mechanism
for crossing below the Gaussian bound that the more expressive matched screen
missed.

[UniSVQ](https://arxiv.org/abs/2606.10520) is likewise a short-vector affine
integer-lattice codebook (`d=4` in its main runs), which is contained as a
reconstruction family by the arbitrary 4-D nonparametric test-channel screen.
Its isolated Qwen result is an absolute MSE/SNR table rather than this pooled
normalized metric (the reported 8.48 dB SNR corresponds to relative error
around 0.142 under the conventional definition), and its final grid tuning
uses RedPajama activations and block-output loss.  The newer
[CubicQuant](https://arxiv.org/abs/2608.06763) is a parametric scalar codebook,
contained by the completed nonparametric scalar screen.  Neither provides an
untested source-specific route below the Gaussian limit.

## 4. Existing embedding/attention weights as a sparse dictionary

The one-atom oracle searched 768 hash-selected 2048-vectors from a held-out
expert against the complete 151,936-row embedding and a 4,096-row attention
Q dictionary. It grants an exact real coefficient for distortion and searches
absolute cosine exactly in blocked CuPy GEMMs.

| Dictionary | Actual energy explained | Random control | Free `s` | Charged side BPW | Required energy | Actual max cosine |
|---|---:|---:|---:|---:|---:|---:|
| Embedding | **1.0463%** | 1.0420% | 0.007587 | 0.016602 | 20.9382% | 0.1426 |
| Attention Q | **0.8412%** | 0.7035% | 0.006094 | 0.013672 | 20.6164% | 0.1607 |

The embedding result is statistically indistinguishable from a random
dictionary. After an 18-bit index and FP16 coefficient its net `s` is
negative. The required uniform cosine is 0.4576, versus the measured mean
0.1021 and random extreme prediction about 0.1110.

Read bandwidth independently disfavors larger sparse supports. With the
current 10/9 residual stream, a one-atom embedding predictor can remain below
2x only if dictionary atoms are independently addressable at at most about
2.224 bpw; at 2.15-bpw atoms and the observed 764/768 uniqueness, the favorable
projected total is about 1.966x. BF16 atoms instead push it above 7x. Two or
more atoms exceed 2x without extreme reuse. The one bandwidth-feasible case
fails the MSE gate by roughly twenty-fold.

This behavior is consistent with sparse regression coding theory: random
sparse superpositions can approach the Gaussian rate-distortion function, but
do not create the source-specific entropy deficit needed to beat it. See the
primary [SPARC lossy-compression result](https://arxiv.org/abs/1401.5272).

## Reproduction

The protocol was serialized before either result was opened:
`protocol_freeze.json`. On the supplied RunPod (RTX 5090, CuPy 14.2), run:

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/breakthrough_redteam/cut_factor_oracle.py \
  --source-dir blind_protocol_v2/unblinded/sources \
  --identities layer18_expert20 layer5_expert18 \
  --max-rank 1380 --starts 2 --alternating-steps 3 \
  --output INT2_Q_C/research/breakthrough_redteam/cut_factor_result.json

/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/breakthrough_redteam/semantic_dictionary_oracle.py \
  --source-dir blind_protocol_v2/unblinded/sources \
  --embedding qwen_weight_cache/tensors/model.embed_tokens.weight.bf16.bin \
  --attention qwen_weight_cache/tensors/model.layers.15.self_attn.q_proj.weight.bf16.bin \
  --identity layer18_expert20 --rows-per-role 256 --block-rows 4096 \
  --output INT2_Q_C/research/breakthrough_redteam/semantic_dictionary_result.json

/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/breakthrough_redteam/verify_results.py \
  --directory INT2_Q_C/research/breakthrough_redteam \
  --source-dir blind_protocol_v2/unblinded/sources \
  --embedding qwen_weight_cache/tensors/model.embed_tokens.weight.bf16.bin \
  --attention qwen_weight_cache/tensors/model.layers.15.self_attn.q_proj.weight.bf16.bin \
  --output INT2_Q_C/research/breakthrough_redteam/verification_receipt.json
```

The verifier independently rehashes every source and dictionary, reconstructs
both rate ledgers, recomputes every pooled cut-factor statistic, and checks the
semantic rate/energy thresholds. Result JSON contains per-matrix curves,
seeds, elapsed times, source hashes, selected-row bindings, and selected-index
hashes.
