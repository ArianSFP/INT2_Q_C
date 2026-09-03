# Independent source review: TACTIC Ramanujan-384 Qwen pilot v0

Date: 2026-09-03

Pinned producer:

- manifest SHA-256: `340ef7f532ab02e03bf04257f3ff07dbc4736bd9e5e96203169603df918e3a8a`
- source root SHA-256: `611bf1b9c822cb90f32a2956e52d8332ef75374186e4acedc958ec3a6c5468ec`
- exact closure: eight members plus `SOURCE_MANIFEST.json`

This is an independent, read-only static review. It did not modify the
producer or any dependency, initialize Python, NumPy, CuPy, CUDA, access a
Qwen or coarse payload, or create a commit. The reviewer did not use the
network. The parent authority reported that the RunPod endpoint became
reachable at TCP level but refused or timed out during the SSH banner; no
remote command executed.

## Disposition

```text
PASS_STATIC_FAIL_CLOSED_QWEN_PILOT_ARCHITECTURE__
HOLD_FINAL_SOURCE_PYTHON_CUPY_CAPABILITY_PAYLOAD_RD_AND_HBM
```

The static architecture satisfies the requested source-only design:

- `TRUSTED_CAPABILITY_SHA256` is literally `None`, and authorization stops
  before the capability path is resolved or opened.
- The capability pins atomic v3, its independent source review, scalable v2,
  its independent source review, actual-coarse v6, and the domain-separated
  coarse-result-auditor closure.
- Future production requires two distinct exact runtime-audit packages. Their
  pinned receipts must be members of those closures and bind audit kind,
  auditor identity, and the audited manifest and source root.
- Source and independently decoded coarse FP32 images are opened only after
  the compiled capability authenticates. They are hash-bound to the frozen
  independent coarse-result audit.
- The sample is fixed in source at sixteen whole 4,096-weight blocks per role.
  The early aperture literally emits, decodes, and source-domain scores every
  representable rank 0 through 14 state, using batched CuPy-oriented solves and
  reconstruction rather than a projected capture.
- The 4,096-replicate deterministic bootstrap requires both pooled and every
  role-owner 5% lower bound to reach `0.32387022205373717`, and requires the
  implied conservative `D<=0.025`, before full-expert work is permitted.
- A survivor writes and replays exactly 1,470,464 bytes, proves exact rate
  `359/144 = 2.493055555...` bpw, verifies the replayed coarse bytes equal the
  authenticated coarse frame, decodes literal fine packets, and performs a
  pooled original-BF16 FP64 rescore.
- Controls remain after the full decoded `D<=0.025` gate: one frozen phase
  destroyer and eight moment-matched Gaussian controls, with the strongest
  control used for the source-specific excess test.
- The page trace records one 4 KiB host-file read for each of 359 packet pages.
  It explicitly disclaims accelerator-HBM evidence and projected layout.

## Conditions and limits

The final hardened producer source was not Python-executed. A 17/17
source-only run occurred before the final authority hardening and is preserved
only as historical mechanism evidence; neither producer nor this review
promotes it to final-source test evidence.

The frozen producer is deliberately non-runnable. A future deployment must
solve the external authority/bootstrap problem, compile the one accepted
capability digest, supply two real non-aliased independent audit closures, and
then run the documented isolated command. This review does not certify such a
future sibling.

The full rescore uses the separately hash-pinned FP32 image already decoded by
the independent coarse auditor. The runner verifies that the replayed
container's coarse bytes equal the authenticated `COARSE.bin`, but it does not
perform a second coarse decode inside the pilot process. This is consistent
with the stated audited-residual design, not standalone container-decoder
evidence.

The fixed deterministic sample plus nonparametric block bootstrap is a severe
compute-admission rule, not a design-based confidence interval over a random
population sample. Passing it permits the literal full-expert test; it is not
itself a Qwen result.

The host page trace excludes the source files, independently decoded coarse
FP32 images, code/model pages, decoder scratch, caches, and GPU traffic. It
therefore cannot establish the final below-2x inference-HBM objective.

No early-aperture outcome, Qwen distortion, Gaussian-control outcome,
`D<=0.025`, `F<=0.8`, portability, universal-SwiGLU-MoE, or inference bandwidth
claim follows from this review.

## Static replay

From this review directory:

```powershell
.\review_static.ps1 -Producer ..\tactic_ramanujan384_qwen_pilot_v0
```

When Python becomes available, the source-closure verifier can be run without
payload access:

```bash
python -I -B verify.py \
  --review . \
  --producer ../tactic_ramanujan384_qwen_pilot_v0 \
  --expected-review-manifest-sha256 REVIEW_MANIFEST_SHA256
```
