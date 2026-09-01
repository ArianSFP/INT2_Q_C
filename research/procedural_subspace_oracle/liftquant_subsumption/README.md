# LiftQuant assessment: already subsumed by binary additive VQ

No new beam/exact experiment was launched because the proposed LiftQuant code
is not a genuinely distinct architecture from the binary arm of the existing
additive-VQ screen.

[LiftQuant](https://arxiv.org/abs/2606.04050) reconstructs a `d`-vector as

\[
\widehat w=Mq,\qquad q\in\{-1,+1\}^{D},\qquad R=D/d.
\]

An additive VQ with `D` binary stages stores arbitrary codeword pairs
`C[j,0]`, `C[j,1]` and reconstructs

\[
\widehat w=\sum_j C[j,b_j].
\]

Define

\[
a=\frac12\sum_j(C[j,0]+C[j,1]),\qquad
M_{:,j}=\frac12(C[j,1]-C[j,0]),\qquad q_j=2b_j-1.
\]

Then the additive reconstruction is exactly `a + Mq`.  Conversely, plain
LiftQuant is obtained with `C[j,0]=-M[:,j]`, `C[j,1]=M[:,j]`, and `a=0`.
Therefore:

- plain LiftQuant is a strict subset of arbitrary binary additive VQ;
- affine LiftQuant is exactly the same code family;
- the existing additive-VQ experiment is more favorable because it also
  grants/charges per-matrix centering, RMS, and a fitted reconstruction gain.

[`verify_liftquant_subsumption.py`](verify_liftquant_subsumption.py) checks the
identity exactly over all 32 assignments of a deterministic integer example,
binds the official source, and rechecks the relevant result identities.

## Existing held-out evidence

The sibling [`../../additive_vq/`](../../additive_vq/) experiment trained
role-conditioned codebooks on five Qwen experts and evaluated the held-out
sixth, with an independently fitted matched-Gaussian pipeline.  It charged
FP16 tables, per-matrix scalars, framing, and index payloads and rehashed all
18 live source matrices.

Its binary rows map directly to LiftQuant with `D = stages`:

| d | D | payload R | physical R | source D | matched Gaussian D | charged matched `s` | favorable 2-SE `s` | codec F | cold read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | 2.25 | 2.250558 | 0.11489658 | 0.11007133 | -0.0315069 | -0.0105307 | 2.60183 | 1.00135x |
| 16 | 36 | 2.25 | 2.252023 | 0.11103331 | 0.10771729 | -0.0238945 | -0.0107898 | 2.51945 | 1.00526x |
| 32 | 72 | 2.25 | 2.257883 | 0.10920263 | 0.10563844 | -0.0318190 | -0.0189833 | 2.49812 | 1.02088x |

All three source/control ratios are worse than one.  Even the deliberately
favorable two-standard-error allowance remains negative, versus the required
`s=0.160964047443681`.  A separate d=8/16 LiftQuant run would therefore
repeat an already-contained family and change only search quality, contrary
to the instruction to run it only if genuinely distinct.

## Official-source binding

The frozen receipt binds the
[official repository](https://github.com/Heliulu/LiftQuant) at commit
`72b3875c770e4579639931fed89dc95e4067edac`:

- `README.md`:
  `0e113b089e293b4e82e07962daf5e5f026d76b65f400d6b9cd2aab05c87dce6b`
- `lattice_generator2.py`:
  `0914967462ec5e76ea27a4b38c7412082b9bbb61f0bef9ef1780178cfbcaaafd`
- `quantize/tmplinear.py`:
  `ca4949d453b147501ed363efda1c56dc2cf3428628414ab719c5ecc5d8b27da9`
- included paper PDF:
  `7065ebbde21fc8e7454aa249ec778ce06b2444f9d9f3f16bb42ad6c526107e01`

The official projection search and decoder use binary signs followed by
multiplication with the learned projection matrix, matching the algebra
above.

## Verification

With a checkout of the pinned official repository:

```bash
/workspace/int2-cupy-venv/bin/python \
  research/procedural_subspace_oracle/liftquant_subsumption/verify_liftquant_subsumption.py \
  --additive-result research/additive_vq/final_xklt/additive_vq_screen_result.json \
  --additive-receipt research/additive_vq/final_xklt/verification_receipt.json \
  --liftquant-root /path/to/LiftQuant
```

The frozen receipt is
[`liftquant_subsumption_receipt.json`](liftquant_subsumption_receipt.json),
lock
`291f1a91522f831302d94cbf93b64356e18f7fd9ad16b81353d7e2ea6cf5b70a`.

## Claim boundary

The prior additive encoder used residual initialization and coordinate
sweeps, not globally exact nearest-hypercube search.  This establishes exact
family containment and a strong, cross-fitted negative screen; it is not an
optimizer-independent converse for every possible projection matrix or beam
width.
