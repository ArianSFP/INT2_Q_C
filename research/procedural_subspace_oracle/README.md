# Procedural/subspace and structured-binary oracle screens

This directory freezes three CPU-only research branches evaluated on the
same pinned 18-matrix Qwen3-30B-A3B MoE panel.  Every branch asks the
rate-relative question required by the project: does Qwen structure lower
MSE relative to an identically optimized, moment-matched Gaussian control by
the required

\[
s=-\tfrac12\log_2(D_{\rm Qwen}/D_{\rm Gaussian})
\ge 0.160964047443681,
\]

equivalently 20% below the Gaussian reference (`F <= 0.8`), while keeping a
physical rate in 2.15--2.5 bpw and expert cold reads below 2x.

## Outcome at a glance

| branch | strongest leakage-safe structural `s` | physical codec `F` | cold read | decision |
|---|---:|---:|---:|---|
| PRG union of subspaces | 0.0014553 bpw | 1.05360 at 2.5 bpw (optimistic coefficient model) | 1.0x | hard kill |
| NanoQuant-style binary factor | 0.0136368 bpw at 2.5 bpw | 3.19136 | 1.0x | hard kill |
| LiftQuant `Mq`, binary additive equivalent | best binary additive rows have negative matched `s` | 2.498--2.602 near 2.25 bpw | 1.001--1.021x | already subsumed; no duplicate run |

The initially exciting rank-764 SVD number `0.75797074295` is explicitly
superseded.  It is the ratio of two discarded-tail energies after granting
the exact, source-specific SVD reconstruction for free; it is not codec
`F=D*2^(2R)`.  The detailed DOF, metric/Jacobian, and rate audit is in
[`nanoquant_binary_factor/README.md`](nanoquant_binary_factor/README.md).

## Directory map

- [`prg/`](prg/) contains the procedural seed-selected union-of-subspaces
  screen, full confirmation, broad smoke screen, frozen source hashes, and an
  independent verifier.
- [`nanoquant_binary_factor/`](nanoquant_binary_factor/) contains the
  continuous spectral hypothesis screen, its formal SVD-leakage
  supersession, a charged discrete LB-ADMM/SVID factor test, exact ledgers,
  and independent verification.
- [`liftquant_subsumption/`](liftquant_subsumption/) proves the algebraic
  containment of LiftQuant in binary additive VQ and binds that conclusion to
  the repository's already-verified cross-expert additive-VQ experiment.

## Shared source seal

- Qwen checkpoint: `Qwen/Qwen3-30B-A3B`, revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Plan lock:
  `99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`
- Plan file:
  `8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868`
- Expert-affine XKLT header:
  `3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538`
- Coverage: 18 matrices, six experts, gate/up/down, 28,311,552 weights.

The result and verification receipts contain all 18 exact BF16 source hashes.
No program in this directory imports CuPy, Torch, a CUDA API, or launches a
GPU subprocess; this branch intentionally remained CPU-only while the shared
GPU encoder was running.

## Claim boundary

These are aggressive early-kill screens of named architecture families, not
universal compression converses.  The negative conclusions are strong enough
to stop these families because even their favorable oracles are far from the
required structural advantage.  They do not rule out a genuinely distinct
semantic ordering, code family, or joint model.
