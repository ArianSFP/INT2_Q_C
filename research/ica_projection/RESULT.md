# Hidden multivariate negentropy / ICA oracle

## Decision

Kill this branch. A repeated block-orthogonal ICA/projection-pursuit transform
does not expose remotely enough hidden non-Gaussianity on the pinned Qwen
panel.

The required rate advantage is

```text
s_required = -0.5 log2(0.8) = 0.16096404744368115 bpw.
```

The most favorable full-panel result gave away the transform and all model
bits, selected and evaluated on the same panel, subtracted a matched-Gaussian
histogram bias, and added diagonal-variance allocation gain. Its best score was
only

```text
s_free = 0.0014536860633339278 bpw
       = 0.9031122703612124% of s_required.
```

That winner was **identity**, not ICA (XKLT coordinates, `d=16`). The best
held-out, rate-matched scalar-RD proxy was

```text
s_LOO = 0.0019931529366581757 bpw
F_LOO = 2^(-2 s_LOO) = 0.9972407171612078
```

at 2.15 bpw on raw `d=64` blocks. This is 1.23825970352513% of the needed rate
advantage. Even adding a deliberately excessive `0.005 bpw` numerical/support
allowance yields `s=0.006993152936658175`, `F=0.9903522723570432`, and a
remaining shortfall of `0.15397089450702298 bpw`. The required factor is
`F<=0.8`.

## What was tested

- Literal 28,311,552-weight / 18-matrix plan, lock
  `99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`.
- Both raw `{Gate, Up, Down.T}` and the checkpoint header's literal FP32
  `{Gate, XKLT0, XKLT1}` coordinates.
- Contiguous repeated orthogonal blocks `d={8,16,32,64}`.
- Identity, KLT, and symmetric orthogonal FastICA/projection pursuit using
  `tanh`, Gaussian, and cubic contrasts, including distinct deterministic
  starts.
- Whole-matrix mean/RMS normalization, granted for free in the initial screen.
- 2,048 deterministic sample vectors per matrix and 384 fit vectors per
  matrix. A 128-bin density over `[-8 sigma,+8 sigma]` measures the exposed
  marginal shape.
- An identically sized Gaussian control passed through the same histogram
  apparatus. Its finite-sample bias is fully credited back to ICA.
- For the two most informative block sizes (`d=16,64`), six
  leave-one-expert-triplet-out fits. Numeric rotations, scales, histogram
  tables, and quantizer symbol probabilities use only the other five experts.
  The contrast/seed choice remains full-panel selected, an intentional bias in
  the candidate's favor.
- Held-out probability cost is cross entropy under the training symbol prior,
  not an entropy refit on test. Entropy-coded uniform scalar-quantizer curves
  are interpolated in log-MSE at coefficient-payload rates 2.15 and 2.5 bpw
  and divided by a separately generated matched-Gaussian curve. This removes
  the scalar quantizer's shaping loss from the reported advantage. Transform
  and table bits remain free in this early-kill score and are accounted below.

No CuPy, torch, CUDA call, or GPU process was used. The runs used NumPy 2.1.2
on CPU and took 12.28 seconds for the complete free-side screen and 22.58
seconds for the bounded held-out confirmation.

## Free-side results

Every bit of transform/table side information is free in this table. The
score is calibrated histogram negentropy plus diagonal-variance allocation
gain.

| Coordinates | d | Selected transform | Free-side gain (bpw) | Histogram part | Variance part |
|---|---:|---|---:|---:|---:|
| raw | 64 | identity | 0.001160466214337583 | 0.001009445320289792 | 0.000151020894047790 |
| raw | 32 | identity | 0.001133896440505784 | 0.001038542961333544 | 0.000095353479172240 |
| raw | 16 | identity | 0.000857997858674478 | 0.000806033611179924 | 0.000051964247494554 |
| raw | 8 | identity | 0.000985283622937616 | 0.000940560875632990 | 0.000044722747304626 |
| XKLT | 64 | cube ICA | 0.001356912541627122 | 0.001287792978987845 | 0.000069119562639277 |
| XKLT | 32 | identity | 0.001286400408757188 | 0.001200472588737789 | 0.000085927820019399 |
| XKLT | 16 | identity | **0.001453686063333928** | 0.001408811728238923 | 0.000044874335095004 |
| XKLT | 8 | cube ICA | 0.001084898000354147 | 0.001034046473222361 | 0.000050851521731786 |

ICA loses to identity in six of eight branches. Its two wins are only tiny
finite-sample-scale movements and remain about two orders of magnitude below
the target.

## Leave-one-expert-out confirmation

`s` is `-0.5 log2(D_Qwen / D_matched-Gaussian)` after each curve is evaluated
at the same rate.

| Coordinates | d | Cross-fit histogram gain | `s` at 2.15 | `F` at 2.15 | `s` at 2.5 | `F` at 2.5 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 64 | 0.000969346550923412 | **0.001993152936658176** | 0.997240717161208 | 0.000938375249284918 | 0.998699981440509 |
| raw | 16 | 0.000603021483664929 | -0.000077778006964212 | 1.000107829025605 | -0.000450365048660901 | 1.000624533467270 |
| XKLT | 64 | 0.000122889669734801 | 0.000237941058920270 | 0.999670198048364 | 0.001775490367888520 | 0.997541674358588 |
| XKLT | 16 | 0.001072433863708650 | -0.004055577472323850 | 1.005638058544210 | -0.000353439184970116 | 1.000490090804397 |

The free-side screen was already more than 110 times short in rate-gain terms,
so `d=8,32` were not subjected to the more expensive outer-fold confirmation.
That is the intended early-stop rule; all four dimensions remain present in
the more favorable screen.

## Side bits and expert reads

Bandwidth is not the reason this architecture fails. A conservative `d=64`
ledger stores a common FP32 rotation, FP32 means/scales, uint16 density and
quantizer tables, and framing: 41,472 bytes total, or `0.01171875 bpw` over the
panel. If the complete common model is reread cold for every expert, on top of
the expert-affine structural `10/9` read ratio, the read amplification is

```text
1.1438145994832043x at 2.15 bpw
1.1392361111111111x at 2.50 bpw.
```

For `d=16`, side cost is `0.0020887586805555555 bpw` and cold read
amplification is `1.1169402051033592x` at 2.15 bpw. All tested layouts are far
below 2x. Charging the side bits would only reduce the already negligible
gain; they were deliberately omitted from the kill score. In particular, the
best raw `d=64` payload result becomes
`s_net=0.0019931529366581757-0.01171875=-0.009725597063341824 bpw`
(`F=1.0135738396688987`) when the common side model is reserved inside a fixed
panel-wide cap. The reported cold-read calculation is even more conservative:
it pessimistically appends the complete common model to a full-rate payload
rather than subtracting the reserved side bytes first.

## Exact source binding

Each physical BF16 file was independently streamed and rehashed on the RunPod
after the experiment. The verifier returned `physical_sources_checked=true`.

| # | Tensor | SHA-256 |
|---:|---|---|
| 0 | `model.layers.5.mlp.experts.18.gate_proj.weight` | `fe4fd2b8438d868a4b118df31f2886d36c2178c93132e5738e64008d1717a51c` |
| 1 | `model.layers.5.mlp.experts.18.up_proj.weight` | `857b57d1d37140bf10dbc582884c73c632f5a58cd1367f342c6903900e2b376b` |
| 2 | `model.layers.5.mlp.experts.18.down_proj.weight` | `8a1d32393816267ff6050d613a541250d73ff44a6ef6b2c43671d5904e1c7fe0` |
| 3 | `model.layers.12.mlp.experts.7.gate_proj.weight` | `eb56b4470fd98d169eba647262da183b66a94569a7a3a0869ac4d3148357abca` |
| 4 | `model.layers.12.mlp.experts.7.up_proj.weight` | `825cded5f7994df0363a4d170461964210ad56a8bfaecaf25b6166a6f5e1f156` |
| 5 | `model.layers.12.mlp.experts.7.down_proj.weight` | `160bbb9013002ff7301245402ff45654ac410f93208de8d42965b9b34b45df18` |
| 6 | `model.layers.18.mlp.experts.20.gate_proj.weight` | `74dbcaca0211e35a73cb18299fe3ff29bae0aeab58beb546f3c85d069b9fbaf6` |
| 7 | `model.layers.18.mlp.experts.20.up_proj.weight` | `8f44135aa4099014c740c99c87c037dad49b623a1e15bdceea1abe7917775b07` |
| 8 | `model.layers.18.mlp.experts.20.down_proj.weight` | `b0910cf461fbae04f904b4329c7149e7df68ed65caf6d8abb1b690966edad064` |
| 9 | `model.layers.28.mlp.experts.83.gate_proj.weight` | `236f15d3fea493b4b5012f3638d82b5e3db5429b952a93a81535d536d83e4867` |
| 10 | `model.layers.28.mlp.experts.83.up_proj.weight` | `95f654526d3de5860112020b170402e9d72c590437f4dd8518d15b1b07f9ff47` |
| 11 | `model.layers.28.mlp.experts.83.down_proj.weight` | `1b64f058bc1b7ceb755d2526b4196cd06c3879120f114ebbd13567e3971757df` |
| 12 | `model.layers.36.mlp.experts.76.gate_proj.weight` | `b77e0bd39b951ad983ffd511347f548cbda00806ab1dd300786df108cae1ad3f` |
| 13 | `model.layers.36.mlp.experts.76.up_proj.weight` | `b227044ca4db76a59086646727808b87f20664ab73e6ada19e106db38675fd26` |
| 14 | `model.layers.36.mlp.experts.76.down_proj.weight` | `442027369ca97e1ab072176a2a05a072329a36a90c62fc67c1946f1edeed4b69` |
| 15 | `model.layers.45.mlp.experts.41.gate_proj.weight` | `8c064b7b0fd4e04dd5c033b4264b2d3fe594ab95ab387d8a06d711e99a47eb9b` |
| 16 | `model.layers.45.mlp.experts.41.up_proj.weight` | `ba210f1528279f64241f3c01ba7cf5a722cdd0b10288448bc8b1c35b00dc4939` |
| 17 | `model.layers.45.mlp.experts.41.down_proj.weight` | `35de3de5ab4838b4777e3c0fe808b8faf3bd0fc24f934185a1a9800521362349` |

## Reproduce and verify

On the RunPod:

```bash
cd /workspace/INT2__compression/ica_projection_oracle
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python3 free_side_screen.py \
  --plan /workspace/INT2__compression/strata_expert_affine_milestone_v1/plan.lock.json \
  --output qwen_ica_free_side_screen.json \
  --dimensions 64,32,16,8 --representations raw,xklt \
  --vectors-per-matrix 2048 --fit-vectors-per-matrix 384 \
  --iterations 16 --tolerance 1e-6 --histogram-bins 128 --histogram-edge 8.0

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python3 ica_projection_oracle.py \
  --plan /workspace/INT2__compression/strata_expert_affine_milestone_v1/plan.lock.json \
  --output qwen_ica_crossfit_confirmation.json \
  --dimensions 64,16 --representations raw,xklt \
  --vectors-per-matrix 2048 --fit-vectors-per-matrix 384 \
  --iterations 16 --tolerance 1e-6 --histogram-bins 128 --histogram-edge 8.0

python3 verify_results.py --check-sources
```

Evidence hashes:

```text
2763988dfd0bc64d2219898aeb72abd52d3938a3257b770ed800d333fa89ad60  ica_projection_oracle.py
ec6b3c480f8230cadaa7aab3ea11625ce205cdacc164884ba2786d10dadc7c97  free_side_screen.py
21d333c918c172818e385708e52114dfc989bfa787874b83b643d7237a8d2ef8  verify_results.py
2bf8eacad3697fd4666993ce628a9f3ecd2ae2009f1e7aec2efdd8dbcacf0829  qwen_ica_free_side_screen.json
2d69b20c3291085f8a1b33de815ce95092dc699d7072fbade79deb8ade657ace  qwen_ica_crossfit_confirmation.json
8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868  plan.lock.json
3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538  header.bin
```

## Claim boundary

This is a strong empirical kill for stationary repeated contiguous
block-orthogonal rotations up to 64 dimensions followed by separable
class-matched coding. It is not a mathematical converse for an arbitrary
2048-dimensional transform, a nonlinear manifold code, deterministic
long-range combinatorial structure, or functional (activation-aware)
equivalence. Those are outside this oracle.
