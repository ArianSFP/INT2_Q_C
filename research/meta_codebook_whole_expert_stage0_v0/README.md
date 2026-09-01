# Whole-expert nonlinear meta-codebook stage 0

## Status

This is a **source-only, not-yet-executed** CuPy experiment. Do not launch it
while another GPU job is active. The runner requires a literal authorization
argument so an incidental invocation cannot open the authenticated panel.

The experiment tests one frozen six-expert feasibility cell. It does not test
all 128 experts in any layer, all 48 layers, or a fresh validation set. A
positive result cannot be described as model-wide generalization.

## Bound panel and frozen split

The only accepted input plan is
`blind_protocol_v2/unblinded/source_hashes.lock.json` beneath the explicit
`--root`:

```text
bytes       46,013
SHA-256     bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23
internal    5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b
schema      int2-qwen-blind-source-finalization-v2
matrices    18
experts     6 triplets
values      28,311,552
```

Every declared `output_relpath` remains resolved relative to the authenticated
lock's parent directory, `blind_protocol_v2/unblinded`. It is not resolved
relative to `--root`, `research/`, or the experiment package.

The source order is six Gate/Up/Down triplets. Down is transposed after load,
so all logical matrices are `768 x 2048`. The split was fixed before this
experiment produced results:

| Slot | Layer/expert | Split |
|---:|---:|---|
| 0 | 5 / 18 | fit |
| 1 | 12 / 7 | holdout |
| 2 | 18 / 20 | fit |
| 3 | 28 / 83 | fit |
| 4 | 36 / 76 | holdout |
| 5 | 45 / 41 | fit |

Expert identity is not an input to the encoder or decoder. The decoder sees a
role one-hot and the fixed coordinate `2*layer/47-1`. FP16 mean and RMS for
every natural 2,048-value row are stored and charged. Opening or adapting to
any other validation data
is outside this package.

## Frozen cell

Every logical matrix is flattened row-major into contiguous eight-weight
vectors. The codec uses:

- `K=32768` codes and a four-dimensional learned latent table;
- a discarded `12 -> 64 -> 64 -> 4` tanh encoder;
- a stored conditional `8 -> 64 -> 64 -> 8` tanh decoder;
- 15 fixed index bits per vector, exactly 1,105,920 index bytes per expert;
- two predeclared seeds, 512 Adam updates, batch 2,048;
- exact nearest decoded-codeword scoring of every vector in both held-out
  experts, tiled as 4,096 vectors by 2,048 codewords.

The choice `K=32768` matches the predeclared high-ranked gate. Tiling keeps
full exact held-out search realistic on an RTX 5090. Peak evaluation-distance
scratch is about 32 MiB per tile; the training assignment matrix is about
256 MiB. Including source, control, parameters and optimizer state, expected
device use is below 3 GiB.
Expected runtime is tens of minutes rather than hours; the runner records the
actual stack, memory and timing.

Training uses only the four fit triplets. Held-out matrices contribute their
charged row mean/RMS and are encoded once for exact scoring; they never update the
codebook, decoder, encoder, optimizer or hyperparameters.

## Gaussian control

Each synthetic row is generated from a fixed matrix seed, centered, and
rescaled to the corresponding source row mean and RMS. Source and
Gaussian runs use the same split, initialization seed, architecture, update
count, batches, FP16 serialization, exact nearest search and rate ledger.
The matched Gaussian result is a diagnostic. It is never transferred to the
Qwen source and cannot promote a failing source result.

## Exact side and rate ledger

The canonical global object is exactly 278,528 bytes:

| Component | Bytes |
|---|---:|
| Global header | 4,096 |
| `32768 x 4` FP16 latent codebook | 262,144 |
| 5,256 FP16 decoder parameters | 10,512 |
| Required zero padding | 1,776 |
| **Total global side** | **278,528** |

Each expert-local frame charges 9,216 bytes for its three matrices' FP16 row
`(mean,RMS)` pairs and a 64-byte header. Keeping row moments local is required
for the cold-read ledger; a decoder cannot read all experts' normalization
tables to reconstruct one expert. With six experts and 28,311,552 weights,
the evidence-panel fixed prefix is

```text
index                   1.8750000000000000 bpw
global side             0.0787037037037037 bpw
expert-local row RLN    0.0156250000000000 bpw
expert-local headers    0.0001085069444444 bpw
fixed prefix            1.9694372106481481 bpw
```

`B=ceil(R*N/8)` is the total physical container size. `B-278,528` is divided
among six expert frames with sizes differing by at most one byte. All index,
header, side and padding bytes are inside `B`.

| Requested R | Physical bytes | Max local frame | Total residual bytes | Residual bpw | Cold bytes | Cold amplification |
|---:|---:|---:|---:|---:|---:|---:|
| 2.15 | 7,608,730 | 1,221,701 | 639,002 | 0.18056290 | 1,503,232 | 1.18540045x |
| 2.30 | 8,139,572 | 1,310,174 | 1,169,844 | 0.33056302 | 1,589,248 | 1.17149747x |
| 2.50 | 8,847,360 | 1,428,139 | 1,877,632 | 0.53056279 | 1,708,032 | 1.15833333x |

Cold bytes pessimistically include the entire global object plus the largest
expert frame rounded up to 4 KiB. No cache residency is assumed.

For completeness, the same serialized global object and expert-local format
give this **arithmetic-only** 128-expert whole-layer projection. It is not an
evidence claim: the six-expert panel cannot establish codebook quality or
generalization across a whole layer.

```text
global side             0.0036892361111111 bpw
expert-local row RLN    0.0156250000000000 bpw
expert-local headers    0.0001085069444444 bpw
index                   1.8750000000000000 bpw
fixed prefix            1.8944227430555556 bpw
required q              0.05788070959170235
```

| Requested R | Whole-layer bytes | Max local frame | Total residual bytes | Residual bpw | Cold bytes | Cold amplification |
|---:|---:|---:|---:|---:|---:|---:|
| 2.15 | 162,319,565 | 1,265,946 | 19,295,437 | 0.25557726 | 1,548,288 | 1.22093023x |
| 2.30 | 173,644,186 | 1,354,420 | 30,620,058 | 0.40557726 | 1,634,304 | 1.20471014x |
| 2.50 | 188,743,680 | 1,472,384 | 45,719,552 | 0.60557726 | 1,753,088 | 1.18888889x |

## Containing early oracle and decisions

For each held-out matrix the stored FP16 codebook and decoder are used to
materialize all 32,768 decoded output-space codewords. Exact nearest
reconstruction gives pooled relative residual energy `q`. Every remaining
physical bit is then
given to an ideal continuous Gaussian residual code. Therefore

```text
F_oracle = q * 2**(2 * 1.9694372106481481)
s_oracle = -0.5 * log2(F_oracle)
required q <= 0.05216397006684782
```

This favorable oracle contains the declared finite first stage plus Gaussian
residual member. It is not a converse for arbitrary non-Gaussian or nonlinear
residual codecs.

The nonlinear decoder does not create a larger reconstruction family than
the `K` decoded eight-vectors it emits. The gate therefore materializes those
vectors, ignores the discarded encoder, and searches them exactly in output
space. Any benefit claimed for the meta-decoder is side sharing/conditioning,
not representational superiority over directly stored decoded vectors. A
direct FP16 table of all conditional outputs would have to be charged before
it could replace the decoder.

[SoftBinary Coding (ICML 2026)](https://arxiv.org/abs/2606.29578) is recorded
only as a distinct follow-on. It is not part of this cell and should be opened
only if this fixed `K=32768` gate survives or exposes a meaningful positive
matched gap.

The exact aggregate rules are:

- **KILL:** even the better of the two predeclared source seeds has pooled
  `s < 0.16096404744368115`, equivalently favorable `F > 0.8`.
- **PROMOTE_TO_FRESH_AUXILIARY_CONFIRMATION_ONLY:** both seeds have pooled
  `s >= 0.18096404744368115`; every held-out expert in both seeds has
  `F <= 0.8`; and both `s_source-s_gaussian` values are positive.
- Anything else is **HOLD_INCONCLUSIVE**.

Promotion is not a finite residual result or target achievement and does not
authorize fresh data. Kill rejects only this exact training cell.

## Source-only verification

This command reads source files and the JSON source lock only. It never opens
any BF16 payload or imports CuPy:

```bash
/usr/bin/python3 -B -I \
  research/meta_codebook_whole_expert_stage0_v0/verify_source.py \
  --root /workspace/INT2__compression
```

## Deferred RunPod execution

Run only after GPU coordination:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
/workspace/int2-cupy-venv/bin/python -B -I \
  research/meta_codebook_whole_expert_stage0_v0/meta_codebook_stage0.py \
  --root /workspace/INT2__compression \
  --output /workspace/meta_codebook_whole_expert_stage0_v0_run \
  --authorization OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_META_CODEBOOK_STAGE0_V0
```

The output directory must be absent or empty. The run writes separately
serialized source/control side objects for both seeds and a canonical
`result.json`; it never downloads data or launches another process.
