# TACTIC Ramanujan-384 Qwen pilot v0

Date: 2026-09-03

Status:

```text
FROZEN_SOURCE_ONLY__COMPILE_TIME_CAPABILITY_PIN_NONE__NO_PAYLOAD_OPENED
```

This package is a fail-closed actual-payload runner for the frozen
Ramanujan-384 fine codec. It does not open model or coarse payload while its
source is built or tested. The production entry begins with:

```text
TRUSTED_CAPABILITY_SHA256 = None
```

and fails before resolving the capability path. A later independently audited
deployment sibling must replace that value with the SHA-256 of one external
capability file sealed before any payload access. Supplying a digest on the
command line is deliberately impossible.

## Frozen dependencies

The external capability must bind exact closures for:

- atomic v3 manifest/root
  `97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1` /
  `5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674`;
- atomic v3 independent-review manifest/root
  `60feb6ae08b3d57df6056e0912759b1e4eb9eb7888c90467cbfd37e72ba97173` /
  `27f422950b7bdd686541677341665fb075295cdfbdd2e1acac3a5c42ce089cd2`;
- scalable v2 manifest/root
  `1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209` /
  `bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495`;
- v2 independent-review manifest/root
  `4ed8c0fe24db072e22aef84791a01ccf637cb337376a389d47119248fd257281` /
  `16ea8dfde5cf7a48552dc7b5a74b209488934b8764e890bf51bb5cd02985cd39`;
- actual-coarse v6 manifest/root
  `31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d` /
  `161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2`;
- coarse result-auditor v1 manifest/root
  `5386571db2a8e828c09368f603b3ccf0ccf3936204e7e06231d5c5798eb9f97f` /
  `59387c67a18bb776cca820e658be998d75f9c3c1a9b7ef5c809e692f78a50742`.

It also pins and opens the audited Qwen coarse result:

- external audit-receipt file SHA-256
  `e03af88a5d33eaca30f935fffc8fcade477219c1be1afebb952428982e4d48e7`;
- `COARSE.bin` SHA-256
  `6c13780bf1494567f91bc73bf6afd8846c6e3326cac329e4d8e3faf48a9051d7`;
- coarse bytes `1,414,656` and rate `307/128` bpw;
- independently rescored coarse `D0=0.036975150060595235`.

Two further exact audit closures are mandatory: a real atomic-v3 CuPy runtime
audit and an independent audit of this pilot runner. Their receipts must be
executed `PASS`, separately pinned, non-dummy, non-self-authored, and authored
by identities distinct from the external capability issuer. They do not exist
in this package, so the runtime hold cannot be mistaken for a result.
Every closure carries its canonical root-field, root-domain bytes, and row-key
order. This explicitly preserves the legacy v3-review root and lowercase
manifest name rather than mis-verifying it as a sorted-key closure. Every
runtime-audit receipt must itself be a manifest member and bind the audit kind,
auditor identity, audited source-manifest SHA-256, and audited source root. The two audit packages,
authorities, and manifest/root/receipt triples must be non-aliased.

## Source-first early-kill aperture

The block sample is code-frozen before source access: 16 complete 4,096-weight
blocks in each of Gate, Up, and transposed Down. The three public arithmetic
progressions in `capability.py` select 48 of 1,152 expert blocks. Neither the
capability issuer nor the runner can change the sample after seeing weights.

For each sampled block, `aperture.py`:

1. forms the residual against the independently decoded and hash-pinned coarse
   FP32 reconstruction;
2. batches the full rank-prefix linear solve for ranks 1..14;
3. canonicalizes binary16 scales and signed int11 coefficients;
4. emits and decodes every representable literal 48-byte rank-0..14 packet;
5. reconstructs all literal packet states in one CuPy einsum;
6. scores exact FP64 residual SSE for every decoded candidate.

There are no per-candidate solves, matmuls, or device-scalar synchronizations.
Invalid nonzero-rank states receive infinite SSE and cannot win.

The decision uses 4,096 deterministic SplitMix64 bootstrap replicates,
stratified by role owner. The 5% lower confidence bounds must satisfy:

```text
pooled capture LCB       >= 0.32387022205373717
minimum role-owner LCB  >= 0.32387022205373717
D upper from pooled LCB <= 0.025
```

where the capture threshold is exactly
`1 - 0.025 / 0.036975150060595235`. If any predicate fails, the runner writes
`HARD_KILL_SOURCE_FIRST_APERTURE` and terminates before the full-expert search,
phase control, or any Gaussian control.

This is intentionally severe. A favourable sample is permission to spend more
compute, not evidence that the full expert passes.

## Survivor path

Only after the aperture survives does the runner:

- search all 384 blocks of all three roles with the pinned scalable-v2 core;
- write and canonically decode a literal `1,470,464`-byte container;
- enforce exact rate `359/144 = 2.493055555...` bpw;
- require the decoded composite coarse segment to equal the audited
  `COARSE.bin` byte-for-byte, decode the fine packets, and add those corrections
  to the separately hash-pinned independent coarse-F32 decode;
- independently rescore original BF16 coordinates with FP64 SSE and energy;
- stop before controls unless the full decoded result has `D<=0.025`;
- run one phase control and all eight frozen moment-matched Gaussian controls;
- record one explicit 4 KiB page read per page, exactly once.

The page trace is an instrumented host-file read of the routed expert packet,
not an inference-HBM measurement. It records `1x` literal file-read
amplification and explicitly keeps `accelerator_hbm_measured=false`.

The pilot does not implement a second coarse decoder in-process. Its coarse
numeric image is the independent decode pinned by the coarse-result audit and
the external capability. Therefore the survivor is evidence for this audited
composite path, not a claim that `COMPOSITE.bin` alone is a standalone decoder.

No projected capture, transferred score, encoder-only reconstruction, or
unscored packet is accepted. A survivor remains merely eligible for a separate
independent result audit.

## Exact RunPod command

The future deployment sibling, after compiling the independently frozen
capability SHA-256, uses:

```bash
python -I -B /workspace/INT2_Q_C/research/tactic_ramanujan384_qwen_pilot_v0/pilot_runner.py \
  --capability /workspace/authority/tactic_ramanujan384_qwen_pilot_v0_capability.json \
  --output-parent /workspace/results \
  --authorization RUN_PINNED_TACTIC_RAMANUJAN384_QWEN_PILOT_V0
```

Running that command from this v0 source freeze fails immediately with:

```text
HOLD: compile-time external capability SHA-256 is None
```

This is the required state until the capability and both independent audit
closures exist.

## Source-only replay

```bash
python -I -B research/tactic_ramanujan384_qwen_pilot_v0/test_source_only.py

python -I -B research/tactic_ramanujan384_qwen_pilot_v0/verify_source.py \
  --package research/tactic_ramanujan384_qwen_pilot_v0 \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --repository-root .
```

The seventeen tests use only small constructed arrays and schema fixtures.
They do not import CuPy, enumerate a model directory, open Qwen, or open the
coarse payload. The literal-rank test uses one synthetic 4,096-value block and
the pinned local v2 source.

The pre-hardening suite passed 17/17 under Python 3.12.13 and NumPy 2.3.5.
The final authority/root and audit-binding repairs could not be re-executed:
this host has no Python, and the provided RunPod refused and then timed out
before a remote command ran. `SOURCE_ONLY_TEST_RESULT.json` records that hold;
the commands above are the exact pending replay.

## Claim boundary

This package is source and source-test evidence only. It is not a Qwen run, an
early-aperture outcome, a full-expert outcome, a Gaussian-control outcome,
`D<=0.025`, `F<=0.8`, or inference-bandwidth evidence. No payload was accessed.
