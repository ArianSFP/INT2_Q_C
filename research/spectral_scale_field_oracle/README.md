# Spectral scale-field oracle

## Decision

**Early kill.** Long-range two-dimensional magnitude fields are far too weak
to meet the same-rate target on the pinned 18-matrix Qwen panel, even when the
field is fitted on the matrix being scored, revealed to the decoder for free,
and followed by ideal continuous Gaussian reverse waterfilling.

The required condition is

```text
F = MSE * 2^(2R) <= 0.8
s = -0.5 log2(F) >= 0.16096404744368115 bpw,
2.15 <= R <= 2.5,
cold expert read < 2x.
```

The strongest deliberately impossible ceiling simultaneously matches every
row-energy, column-energy, and 16x32 tile-energy marginal with a
source-specific multiplicative field. Its result is:

| Quantity | Exact result |
|---|---:|
| Representation | source-derived Up/Down XKLT |
| Physical rate | `2.5 bpw` |
| Free source-leaky relative MSE | `0.029678913700195278` |
| Free source-leaky `F` | **`0.9497252384062489`** |
| Free source-leaky `s` | **`0.0372089509973228 bpw`** |
| Fraction of required `s` | `23.116311740571532%` |
| Remaining `s` gap | `0.12375509644635835 bpw` |
| Cold expert read amplification | `1.0389067568914487x` |
| Matched-Gaussian structural `s` | `0.018541352011894146 bpw` |

At 2.5 bpw the strict MSE threshold is `0.025`; the free oracle remains
`18.7156548007811%` above it. This is a stop result, not a candidate codec.

Once physical side bytes are charged inside the rate, the best absolute row
is merely a global per-matrix scale field at 2.15 bpw:

| Quantity | Exact result |
|---|---:|
| Relative MSE | `0.049663860798126354` |
| `F` | **`0.9782941562680455`** |
| `s` | **`0.01582988541668536 bpw`** |
| Total side information | `5,184 bits` (`648 bytes`) |
| Cold expert read amplification | `1.0428780916078009x` |
| Matched-Gaussian structural `s` | approximately zero (`4.97e-13 bpw`) |

The charged global advantage comes from ideal rate allocation across matrices
with different energies. It is not Qwen-specific structure: all four
moment-matched Gaussian controls reproduce it.

## Family-by-family ceiling

Each row is the best absolute `F` for that family. “Free” means the exact
source-derived field, model choice, factors, and XKLT angle are all supplied
to the decoder without consuming a bit. The charged result keeps the total
physical rate fixed and subtracts every side bit before waterfilling the
coefficient payload.

| Family | Best free `F` / `s` | Best charged `F` / `s` | Best matched structural `s` |
|---|---:|---:|---:|
| Global matrix scale | `0.9780458595` / `0.0160129909` | `0.9782941563` / `0.0158298854` | `~0` |
| Row x column separable | `0.9516555089` / `0.0357443350` | `0.9904594258` / `0.0069151093` | `0.0184342498` |
| Row x column x tile IPF | **`0.9497252384` / `0.0372089510`** | `1.0322133618` / `-0.0228706056` | **`0.0185413520`** |
| Exact 16x32 tile field | `0.9747570387` / `0.0184427138` | `1.0181716376` / `-0.0129903917` | `0.0010224010` |
| Low-rank log variance | `0.9747570387` / `0.0184427138` | `0.9788165579` / `0.0154447941` | `0.0010224010` free; `0.0006787003` charged |
| Analytic DCT log field | `0.9747570387` / `0.0184427138` | `0.9783191793` / `0.0158114348` | `0.0010224010` free |
| Analytic Haar log field | `0.9747570387` / `0.0184427138` | `0.9783079524` / `0.0158197129` | `0.0010224010` free |

The full-rank low-rank/DCT/Haar rows all reproduce the exact 48x64 tile map,
which is why their free ceilings coincide. The IPF union is stronger: it uses
5,888 source-derived factors per matrix and iteratively matches all three sets
of long-range energy marginals.

## What was tested

All 18 BF16 sources are authenticated against the pinned
`Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Down projections are transposed
to a common 768x2048 orientation. Both raw matrices and a source-derived
two-channel Up/Down KLT are scored.

The candidate set contains 28 models:

- a global matrix variance;
- an exact separable row x column variance field;
- a multiplicative row x column x tile field fitted by 16 IPF rounds;
- a free exact 48x64 field of 16x32 tile energies;
- ranks `1, 2, 4, 8, 16, 32, 48` of the log tile-energy matrix;
- eleven fixed DCT-II low-frequency rectangles from `1x2` through `48x64`;
- six fixed Haar/piecewise-constant grids from `1x1` through `48x64`; and
- physical rates `2.15`, `2.30`, and `2.50` bpw.

For a fitted variance field `v_i`, the oracle solves the exact continuous
reverse-waterfill equations

```text
r_i = 0.5 * max(log2(v_i / theta), 0)
D   = sum_i min(v_i, theta) / sum_i v_i,
```

with one panel-wide water level. Expert payload sizes are then computed from
those actual allocated rates. Across all 336 scored rows, the worst cold read
amplification is only `1.0452739073092303x`; locality is not the failure mode.

## Matched-Gaussian control

Four independent PCG64DXSM controls are generated for every source matrix and
representation. Each is exactly centered and rescaled to the source matrix's
FP64 centered energy. The controls then receive the same leaky fit, candidate
grid, ideal waterfill, side charge, and model selection.

For the best free IPF row, the controls give mean MSE
`0.0304516624575689`; the Qwen/control ratio is `0.974623757949164`, or only
`0.018541352011894146 bpw` of matched structural advantage. Thus the result is
not being driven by finite-sample variance fitting or by heterogeneity that an
iid moment-matched control can reproduce.

## Physical and read ledger

All scale information is private to and colocated with its expert. Shared DCT
and Haar bases are analytic and require zero stored bytes. The charged screen
uses:

- a 512-bit header per expert;
- a 64-bit header, FP16 mean, FP16 global scale, and 16-bit model ID per
  matrix;
- FP16 row/column vectors, tile coefficients, DCT coefficients, or low-rank
  factors as applicable; and
- one FP16 XKLT angle per expert.

The coefficient budget is the requested physical rate minus all side bits.
The exact row/column field costs `816,192` panel bits (`0.028828938802083332
bpw`). The stronger IPF union costs `1,700,928` bits
(`0.060078938802083336 bpw`). Their continuous field values are still treated
as exact after paying only the FP16-sized ledger, an optimism in favor of the
hypothesis.

## Relationship to earlier branches

This is not a repeat of the conditional hyperprior audit. That branch
entropy-coded residual scales for short 2,048-value groups and measured
held-out Gaussian NLL gain. This branch asks whether long-range row/column and
smooth 2-D variance geometry can lower ideal same-rate source MSE. The exact
tile row is retained only as a deliberately favorable ceiling.

It also does not repeat the NanoQuant binary-factor test. NanoQuant's oracle
used free iterative row/column equilibration as a preconditioner before
examining an SVD tail or a discrete binary factor. Here the scale field itself
drives the rate allocation; no low-rank weight reconstruction or binary
factor receives credit.

## Reproduction

The completed RunPod command was CPU-only and restricted BLAS to two threads:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/spectral_scale_field_oracle/spectral_scale_field_oracle.py \
  --source-lock /workspace/INT2__compression/blind_protocol_v2/unblinded/source_hashes.lock.json \
  --source-root /workspace/INT2__compression/blind_protocol_v2/unblinded \
  --output /workspace/INT2__compression/spectral_scale_field_oracle/spectral_scale_field_result.json \
  --control-replicates 4
```

Full source and receipt verification on the pod:

```bash
python verify_spectral_scale_field_result.py spectral_scale_field_result.json \
  --source-lock /workspace/INT2__compression/blind_protocol_v2/unblinded/source_hashes.lock.json \
  --source-root /workspace/INT2__compression/blind_protocol_v2/unblinded
```

The verifier independently checks the result seal, algorithm hash, all 336
`F`/`s` identities, matched-control ratios, side-rate closure, per-expert read
ledger, selected best rows, decision, source lock, and all 18 source hashes.

## Artifacts and identities

- `spectral_scale_field_oracle.py`: reproducible CPU oracle; SHA-256
  `ce9bfae53e0d4b043c7feb8dd85b125540e00dcce4121c83585aa336a55439bd`.
- `spectral_scale_field_result.json`: complete 336-row evidence; SHA-256
  `0782577a0daf458f2d7dd3ac78f89d57c72596230634ccb4494ed9ada5c5b9b9`.
- Result internal lock:
  `95341d0275bc737dfe84980d32d0e9922dc49b965e7881ca8327efae5f0c27f5`.
- `verify_spectral_scale_field_result.py`: dependency-free verifier; SHA-256
  `aa4cfd03ad045db37288ba3616805eeb1ab70431ec621ed9715ae05fc4b9558e`.
- Pinned source-lock internal SHA-256:
  `5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b`.

## Claim boundary

This result rejects the tested separable, multiplicative marginal-matching,
low-rank-log, low-frequency DCT, and Haar variance-allocation models as a path
to `F <= 0.8` on this panel. It is not a lower bound for arbitrary nonlinear,
activation-aware, or cross-expert codecs. No reconstructed-weight MSE or
deployable bitstream is claimed; the favorable oracle failed too early to
justify either.
