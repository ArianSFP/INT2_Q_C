# Canonical Haar/Stiefel orientation-entropy gate

## Decision

**Hard kill for this architecture family.**  Qwen expert orientations do have
a measurable held-out departure from Haar measure, but it is approximately
four orders of magnitude too small to close the remaining rate-relative gap.
The pinned 18 source matrices were not opened and the GPU path was not
launched.

The fixed auxiliary gate used all 16 leakage-clean Layer-15 experts currently
available in `qwen_weight_cache/rd_structure_diag_cross_expert`.  Each fold
removed one whole expert—both Up and Down—from fitting.  The result was:

| Quantity | Result | Required |
|---|---:|---:|
| Raw Qwen held-out ACG gain | `0.00000787498` bpw | — |
| Qwen-minus-Haar mean | `0.00001141609` bpw | — |
| Qwen-minus-moment-Gaussian mean | `0.00001136468` bpw | — |
| Best free-table three-SE upper bound | **`0.00001994733` bpw** | `0.11356063457` nested |
| 512-byte table rate on the pinned panel | `0.00014467593` bpw | included |
| Charged conservative lower bound | `-0.00014189390` bpw | `0.16096404744` standalone |
| Charged optimistic upper bound | `-0.00012472860` bpw | `0.11356063457` nested |

Even giving the table away, the optimistic upper bound supplies only
`0.017565%` of the incremental gain needed by the best existing composite.
Applying that uncharged upper bound optimistically to its `F=0.9363976210`
would yield only `F=0.9363717273`, not `F<=0.8`.  This conclusion does not
depend on the table charge.

The corresponding deliberately favourable high-rate RD envelope is equally
decisive:

| Rate | Gaussian MSE | Standalone, free-table upper | Existing composite plus free-table upper | Target |
|---:|---:|---:|---:|---:|
| 2.15 | `0.0507657748` | `0.0507643710` | `0.0475356362` | `0.0406126198` |
| 2.25 | `0.0441941738` | `0.0441929517` | `0.0413821749` | `0.0353553391` |
| 2.50 | `0.0312500000` | `0.0312491359` | `0.0292616165` | `0.0250000000` |

This conversion grants ideal high-resolution entropy-constrained
quantization, no chart-curvature loss, and no finite-cell or arithmetic-coder
penalty.  It is an upper opportunity envelope, not achieved finite-rate MSE.

## What is new and what is not

The existing `stiefel_gram_oracle` already exhausts the useful energy/DOF split
between a row-Stiefel factor and its symmetric Gram normal.  The
`composite_superoracle` already nests that split with role KLT and STRATA, and
the `polar_normal_predictor` tests the omitted symmetric normal field.  Those
gains are not counted again here.

This gate tests a genuinely different hypothesis: after the polar/QR energy
split, does the *orientation measure itself* have decoder-learnable entropy
below Haar?  Neither the scalar neural-flow screens nor the earlier continuous
Stiefel waterfills measured that intrinsic density deficit.

The result closes one specific, low-side-information family: smooth
role-conditioned diagonal angular-central-Gaussian (ACG) models in a fixed
canonical Householder frame.  It is not a converse for arbitrary nonlinear
manifold codes.

## Canonical source-decodable chart

For each canonical `W` of shape `768 x 2048`, apply raw Householder QR to the
tall transpose:

```text
W.T = Q R,      Q.T Q = I.
```

LAPACK's deterministic reflector convention is

```text
beta = -copysign(||x||, x0)
tau  = 1 + |x0| / ||x||.
```

At reflector `j`, the residual direction lies on
`S^(2047-j)`.  Its coordinates are recovered from the raw reflector as

```text
q0     = -sign(beta) (tau - 1)
q_tail = -sign(beta) tau v_tail.
```

The ordered reflectors reconstruct `Q`; the stored upper triangle reconstructs
`R`; hence their product reconstructs the source matrix.  A literal inverse is
tested in `test_haar_manifold_entropy.py`.

The continuous degrees of freedom close exactly:

```text
Householder/Stiefel = sum(j=0..767) (2048-j-1) = 1,277,568
upper triangle      = 768*769/2                 =   295,296
total                                               1,572,864
```

No source-selected basis, chart, permutation, or target statistic is supplied
for free.

## Why the Jacobian does not create a fake gain

Euclidean NLLs of Householder angles are not comparable without their chart
Jacobians.  This experiment never uses such an NLL.  It scores an ACG density
directly with respect to normalized Haar measure:

```text
log(p_ACG(q) / p_Haar(q))
  = -0.5 log|A| - 0.5 d log(q.T A^-1 q).
```

For stereographic coordinates `y` on a sphere of dimension `p`, the common
measure is

```text
dmu(q) = Area(S^p)^-1 (2 / (1 + ||y||^2))^p dy.
```

That factor is present in both ACG and Haar and cancels exactly in the ratio.
The full QR change of variables has radial factor

```text
dW = constant * product_j |R_jj|^(2048-j-1) dmu_Stiefel(Q) dR.
```

This gate claims only the marginal orientation saving relative to Haar; it
does not relabel the QR radial Jacobian as compression.  Separate coding of
`Q` and `R` is source-decodable but can leave Q–R dependence unused, making the
gate favourable rather than source-leaky.

## Fixed held-out model and controls

Each reflector's squared direction coordinates are accumulated into 16 fixed
relative-coordinate bins.  The 768 reflectors are divided into eight fixed
bands.  For each of two decoder-known role classes—Input (Up in the auxiliary
panel, shared by Gate/Up if promoted) and Down—the model fits a diagonal ACG
shape constant inside each of the `8 x 16` cells.

There are exactly 256 FP16 parameters, or 512 bytes.  Every leave-one-expert-out
fold fits them from the other 15 experts, rounds them to FP16, and then scores
the excluded Up and Down jointly.  Resolution, ridge, precision, confidence
multiplier, and controls were fixed before evaluation; there is no test-set
model selection.

Two controls use the identical raw-QR and fitting path:

1. an iid Gaussian matrix adjusted to each source's exact mean and centered
   energy before its final FP32 representation; and
2. an independent zero-mean iid Gaussian matrix, whose QR orientation is Haar.

The decision uses the worse of the paired Qwen-minus-control bounds.  The
three-standard-error lower bound must reach `0.16096404744` bpw standalone, or
the optimistic `0.11356063457` bpw incremental threshold implied by the
existing composite.  The latter is only a promotion threshold: raw-role gain
would have required a direct role-KLT nested recomputation before it could be
added to any pinned result.

## Exact physical and read ledger

The proposed layout has one 4 KiB global prefix, including the 512-byte FP16
ACG table, followed by six independently decodable contiguous expert frames.
Each frame reserves 160 bytes for its header, stream directory, and CRC; its
two variable streams carry Householder-orientation and upper-triangular
quantization indices.  All bytes are inside the physical cap.

| Requested rate | Actual physical rate | Container bytes | Max cold exact | Max cold 4 KiB |
|---:|---:|---:|---:|---:|
| 2.15 | `2.1499998305` | `7,608,729` | `1.0026918x` | `1.0077520x` |
| 2.25 | `2.2500000000` | `7,962,624` | `1.0025725x` | `1.0061728x` |
| 2.50 | `2.5000000000` | `8,847,360` | `1.0023153x` | `1.0055556x` |

These are exact compressed-object byte/page ledgers.  They are not measured
HBM traffic.  A surviving codec would have to fuse reflector reconstruction
with expert GEMM consumption to avoid materializing BF16 weights.  Because the
entropy gate failed, no finite index stream was emitted and no MSE is claimed.

## CuPy production path and early stop

`haar_manifold_entropy.py --backend cupy` routes raw QR, reflector-coordinate
recovery, squared-energy aggregation, and prefix sums through CuPy/cuSOLVER.
The installed RunPod CuPy implementation explicitly supports raw QR.  The
default is NumPy, and importing CuPy is deferred until the flag is selected.

The sealed initialization gate ran on CPU in 14.30 seconds.  Its optimistic upper
bound was `5,693x` smaller than the nested requirement, so the predeclared
early-stop rule fired.  No GPU context and no pinned source payload were
opened.

## Reproduction

Run the source-blind auxiliary gate on the provided RunPod:

```bash
cd /workspace/INT2__compression
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/haar_manifold_entropy/haar_manifold_entropy.py \
  --aux-dir qwen_weight_cache/rd_structure_diag_cross_expert \
  --target-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --composite-result INT2_Q_C/research/composite_superoracle/result.json \
  --output INT2_Q_C/research/haar_manifold_entropy/result.json \
  --backend numpy
```

Verify bindings, arithmetic, tables, thresholds, and byte/page ledgers:

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/haar_manifold_entropy/verify_result.py \
  --result INT2_Q_C/research/haar_manifold_entropy/result.json \
  --aux-dir qwen_weight_cache/rd_structure_diag_cross_expert \
  --target-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --composite-result INT2_Q_C/research/composite_superoracle/result.json \
  --output INT2_Q_C/research/haar_manifold_entropy/verification_receipt.json
```

Run the NumPy-only chart/ACG/ledger tests:

```bash
cd /workspace/INT2__compression/INT2_Q_C
/workspace/int2-cupy-venv/bin/python -m unittest discover \
  -s research/haar_manifold_entropy -p 'test_*.py' -v
```

The verifier passed 20 independent binding/arithmetic checks and all eight
unit tests passed.  `result.json` binds all 32 auxiliary BF16 files, the pinned
source lock, the parent composite artifact, and the exact executing script.

## Artifact boundary

This is a rigorous negative architecture gate, not a compressed checkpoint.
It supports the statement that the tested canonical low-parameter ACG
orientation model cannot provide a meaningful rate advantage on this
auxiliary Qwen panel.  It does not establish that all manifold entropy models,
all non-diagonal conditionals, or all task-aware codecs must fail.
