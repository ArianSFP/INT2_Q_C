# MOSAIC-Q execution checkpoint — 2026-09-03

## Objective and fixed acceptance test

The research objective remains a universal SwiGLU-MoE post-training codec,
not a Qwen-specific reference or ancestry code. A successful physical artifact
must satisfy all of the following on original BF16 source weights:

```text
2.15 <= physical rate <= 2.5 bpw
F = relative_MSE * 2^(2 * physical_rate) <= 0.8
maximum routed expert cold-file reads < 2x
independent causal decode and decoded-byte FP64 rescore
all headers, models, transforms, padding, and shared pages charged
```

At the current finite 2.5-bpw baseline,

```text
D0 = 0.030902167403153148
F0 = 0.9888693569009007
target D = 0.025
required direct MSE reduction = 19.0995257%
rate-equivalent source advantage = 0.15288996696 bpw
```

For the real `307/128 = 2.3984375`-bpw TACTIC coarse artifact,

```text
coarse D0 = 0.036975150060595235
required coarse-residual capture for D <= 0.025 = 0.32387022205373717
```

The second number, not the finite-baseline 19.10% number, is the promotion
threshold for a refinement of that coarse artifact.

## Distance to the -10% and -20% checkpoints

Two ledgers must remain separate:

| Ledger | Current F | Below Gaussian | Further MSE reduction to F=0.9 | Further MSE reduction to F=0.8 |
|---|---:|---:|---:|---:|
| Finite, read-compliant baseline | 0.9888693569009007 | 1.1130643099% | 8.9869664057% | 19.0995256940% |
| Strongest honest ideal structural composite (not finite) | 0.93639762 | 6.360238% | 3.8869833949% | 14.5662074622% |
| Mixed-role VORPAL quality reference (not a MoE checkpoint) | 0.9362307709 | 6.37692291% | 3.8698547437% | 14.5509819944% |

The VORPAL number is not a universal SwiGLU-MoE checkpoint: it is a mixed-role
development aggregate, much of its gain comes from attention V, its three
expert roles pool to approximately `F=1`, and its audited global read
amplification is above the `<2x` constraint.  It is only a measured codec-quality
reference.  The ideal structural composite shows that roughly 6.36% below
Gaussian has been exposed mathematically, but it is not an emitted or deployable
codec.  The read-compliant MoE ledger is therefore 8.8869
percentage points short of 10% below Gaussian and 18.8869 points short of 20%
below Gaussian.  The quality-only ledger is 3.6231 and 13.6231 points short,
respectively.

All future execution is restricted to the local RTX 3060 at the user's request.
The sealed remote capability is retained unused; RunPod is no longer part of
the execution plan.

## Continuation update after runtime recovery

The RunPod recovered and the supplied endpoint was authenticated as an NVIDIA
GeForce RTX 5090 with CuPy 14.2.0.  The local RTX 3060 lane was also made
usable through a process-local, content-pinned Windows runtime workaround and
a fresh task-specific CuPy cache.  Local execution is restricted to bounded
mechanism/preflight work unless a separately reviewed payload capability is
issued.

Two important payload apertures have since produced auditable negative results:

1. A flat same-layer common/private label model was evaluated on layer 15
   experts `0,8,...,120`, Up and `Down.T`: 50,331,648 authenticated Qwen
   weights.  Its deliberately favorable quaternary common-label saving was
   only `0.04703678191314046` Up/Down bpw, versus the necessary
   `0.22933495044437175` Up/Down-bpw threshold.  Once the common and private
   descriptions were charged, the best gain was
   `-0.01413604560380873` bpw.  A binary common/private page projection did
   stay below the read cap (`1.5919152753x` at 2.15 bpw and
   `1.7676116101x` at 2.5 bpw), so the layout is feasible but the flat source
   model is not useful.  This does not close clustered or hierarchical
   same-layer conditional entropy, flexible joint label selection, Gate
   coupling, or lossy Gray-Wyner reconstruction.
2. Exact independent GF(2) recurrences over existing STRATA selected-SC
   decisions hard-killed at the first 4 KiB page crossing.  The exact
   monotone lower bound was 8,306,688 bytes versus a target maximum of
   8,302,592 bytes, giving rate `169/72 = 2.3472222222` bpw and
   unchanged-reconstruction `F = 0.800124431463777`.  All expert read lower
   bounds remained below `2x`.  This is not a practical one-page near miss:
   only 6,998 of 30,938 chunks had been charged, while every unseen chunk was
   optimistically granted zero payload.  Removing one page of overhead would
   only postpone the same monotone crossing.  The result is a decisive
   negative for the independent-chunk grammar; it does not close a
   fundamentally different cross-chunk, joint-level, or label-flexible
   algebraic code.

The first authorized Ramanujan-384 Qwen production attempt consumed its
one-use claim but failed before scoring the first Gate aperture.  CuPy 14.2.0
rejected a NumPy-style `lexsort(..., axis=1)` call.  The output contained only
the 256-byte claim and no result child, so this is
`FAILED_RUNTIME_NO_SCIENTIFIC_RESULT`, not evidence for or against the
Ramanujan source hypothesis.  Independent failure evidence and exact replay
verification are preserved under:

```text
research/tactic_ramanujan384_qwen_pilot_v1_r3_failure_evidence_runpod_20260903
research/tactic_ramanujan384_qwen_pilot_v1_r3_failure_evidence_independent_verification_20260903
research/tactic_ramanujan384_qwen_pilot_v1_r3_independent_failure_audit_20260903
```

A separately named CuPy-compatible successor replaces the two-key sort with a
stable primary-key `argsort`; the secondary rank is already the unique public
order `0..383`.  Source-free RTX-5090 parity covered all 240 production-shape
candidates, forced and random ties, packet replay, and the downstream 16-block
stream.  Its independently authorized r2 detached preflight was executed once
and failed before NumPy/CuPy import or GPU science: the exact authorization
mandated the venv `bin/python` symlink while the detached auditor required
`sys.executable` itself to be a canonical non-link.  The r2 authority is
consumed and cannot be reused.  The independent failure audit is sealed at
manifest `1cd5015873440c9389f0a3514f486349de8d196862d6ba48caaba734746a3ed3`
and root `6c0bd548303091e6151be3185c1cb58c6749dacb23c1dc57a2b90758047d6310`.
A separately named symlink-aware r3 auditor/deployment is required; there is
still no Ramanujan Qwen result.

That r3 successor has now been independently reviewed and remains blocked
before staging.  The sealed review reproduced an intermediate-symlink
retargeting bypass, found that neither the literal link target nor resolved
interpreter path was pinned before execution, found no durable one-use marker
before fallible checks, and observed that all four POSIX symlink tests were
skipped.  It therefore grants zero RunPod, capability, or Qwen authority.  Its
review manifest is
`70703675717f41ed62b14a35481aa909b1c519e05ef80b3bc7f3a1b019064a53`
with source root
`628b82ea5f5983f6f34cc29c5328fc9056b3c792e84bfb229197fd09019720cf`.
Ramanujan remains scientifically untested, but it is no longer the next
payload branch until those execution-boundary defects are repaired.

The clustered same-layer CBIB-1 successor tests equal groups of
`{2,4,8,16}` experts with eight-fold cross-fitting and a learned two-state
product latent.  Its original and r1 deployments are both infrastructure
blocks with zero Qwen invocations.  The r1 package closed the NumPy/native
binary pinning defect, but its source-free preflight indexed CPU count fields
that the frozen evaluator does not return and failed to compare training
hard-EM assignments.  The separately sealed r2 repair correctly reconstructed
the counts and compared both training and held-out assignments, but its exact
source-free fixture had no candidate satisfying the source threshold and the
strict read bound simultaneously.  The independently reproduced group-size-2
case was source-favorable (`0.42478568491610225` gross bpw) but reached
`2.0413268581241213x` at 2.15 bpw and `2.281655815896668x` at 2.5 bpw;
groups 4, 8 and 16 were both weaker and over the read cap.  The hostile review
therefore authorized zero RTX-5090 preflights and zero Qwen invocations.  A
separately named r3 repair changes only the payload-free mechanism fixture and
charges its exact 256-byte scale ledger; the production evaluator remains
byte-identical to r2.  The r3 source and independent review are sealed at
manifests `5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f`
and `9465a1d6c9ffdb7553721e1a470b2ab485dcb66d8d32e7b21b605e9ff99421e2`.
The one permitted source-free RTX-5090 preflight completed with
`PASS_PRODUCTION_GEOMETRY_FULL_CPU_CUPY_PARITY`: 240 models, 3,932,160
held-out and 27,525,120 training assignments, 960 independently reconstructed
count arrays, all eight controls, and exact quantizer labels/scales were
checked.  Maximum floating deltas were below `1.9e-9`; the receipt is
`38a4eb497983aa8b5a559fa96fcbcbb11dc77cdd78dabfff9d2d4d06c5bf1913`.
It accessed no Qwen payload and consumed its sole attempt.  Its exact evidence
has now passed a detached independent audit.  The audit manifest is
`78476def014acf8053102709b906db5f78139c99eed989b627acf3d9957c8022`,
its source root is
`3567fbc12a932f6af3a099aee0bcb1e7a4005ca3942fe692778784a991890f74`,
and its disposition is
`PASS_AUTHORIZED_SINGLE_PREFLIGHT_RECEIPT_INDEPENDENTLY_AUDITED`.  This proves
the source-free implementation parity only; it does not predict Qwen entropy.

Three separately named local RTX-3060 production capabilities followed.  r1
was consumed before child launch, CUDA initialization, or payload access when
its fresh CuPy cache parent was missing.  r2 initialized CuPy/CUDA and queried
the GPU, then rejected CuPy's 19-byte Windows UUID buffer before the inner
claim or any Qwen access.  Its independent failure audit passes at manifest
`aedca5c3ef06fa97b76cfbfde9e59d704e466faf439aea085b89ed7568250b24`
and source root
`55028d6a591f43507c93eb22fa734c05f5b3d145de27c909bdc766de14cccba3`.
r3 authenticated the first 16 UUID bytes, recorded the three trailing bytes,
and completed the Qwen computation.  Its detached result audit passes with
verdict
`PASS_COMPLETED_CHILD_RESULT_WITH_HARMLESS_STDERR_WARNING__HARD_KILL_CBIB_FIXED_LABEL`,
manifest `c200effe602dcb4fb87a84787ebed0acc35cc5e62c44de7e890da67107822ec5`,
and root `6a23ea1d1a8760bb25eb3633d4943fe27f5e0bad632e1044c194b7bf553c9ada`.

The external `ArianSFP/INT2_Q` VORPAL repository was also audited directly.
Its source-free physical verifiers pass, but the result is not a missing MoE
breakthrough.  Base VORPAL has `F=0.9362307709` at `2.4859493256` bpw; the
expert Gate/Up/Down roles pooled are approximately Gaussian while much of the
favorable aggregate comes from attention V.  Its global variance pooling
crosses tensor, role and layer boundaries: published expert-role blocks touch
4--22 arithmetic chunks, giving a favorable lower-bound mean read near
`9.04x`.  The procedural role-joint SPARC stages are also rate-relative
adverse: every published 242-byte stage removes less SSE than its rate penalty,
so the optimal published global prefix uses zero stages.  Only the small
A64-to-A128 overload upgrade, exact byte-Pareto selector and coordinate-pulse
prefix remain bounded engineering candidates.  Full evidence is in
`docs/INT2_Q_VORPAL_REPOSITORY_AUDIT_2026-09-03.md`.

## Frozen Ramanujan-384 Qwen pilot

The non-dyadic Ramanujan branch now has a fail-closed, source-first Qwen pilot:

```text
research/tactic_ramanujan384_qwen_pilot_v0
```

Producer closure:

```text
SOURCE_MANIFEST.json SHA-256
  340ef7f532ab02e03bf04257f3ff07dbc4736bd9e5e96203169603df918e3a8a
canonical source root
  611bf1b9c822cb90f32a2956e52d8332ef75374186e4acedc958ec3a6c5468ec
8 members plus manifest
```

The production capability digest is intentionally compiled as `None`. The
runner therefore stops before resolving, statting, hashing, or opening any
capability, Qwen weight, or coarse artifact. A deployment sibling may set the
digest only after an external capability binds the complete producer,
dependency, runtime-audit, Qwen-source, coarse-packet, and coarse-audit
closures.

The sample aperture is fixed before source access:

- 16 complete 4,096-value blocks from Gate;
- 16 complete blocks from Up;
- 16 complete blocks from transposed Down;
- literal rank-0 through rank-14 48-byte refinement packets;
- encode/decode before FP64 source-domain scoring;
- 4,096 deterministic role-stratified bootstrap replicates.

The pilot hard-kills unless both the pooled and minimum-role 95% lower
confidence bounds on residual capture reach `0.32387022205373717`, and the
implied upper bound on `D` reaches `0.025`. It performs no full-expert search
and no Gaussian controls after an aperture miss. This makes an adverse result
cheap and prevents a favourable pooled average from hiding a failed matrix
role.

If the aperture survives, the frozen survivor path emits one literal object:

```text
coarse bytes          1,414,656
fine bytes               55,296
container header            512
total bytes            1,470,464
exact rate               359/144 = 2.4930555555555556 bpw
4 KiB pages                   359
```

It then reconstructs from decoded coarse and fine bytes, rescoring every
original BF16 coordinate in FP64. Only a full decoded `D <= 0.025` result is
allowed to launch the one phase-destruction and eight moment-matched Gaussian
controls. The page trace reads each host-file page once, but is explicitly not
reported as accelerator-HBM evidence.

## Independent static review

The independent review is frozen at:

```text
research/tactic_ramanujan384_qwen_pilot_v0_independent_source_review_20260903
```

Review closure:

```text
SOURCE_MANIFEST.json SHA-256
  bfb280c72b92f4c1b53ebade125741c75cd5cf5500437f61d11276629c504cb4
canonical source root
  3079a43995815b8852673e86d95c31965d0d5e98af7ffb56ae24dab259ddb677
5 members plus manifest
```

The PowerShell static replay passed. Its disposition is:

```text
PASS_STATIC_FAIL_CLOSED_QWEN_PILOT_ARCHITECTURE__
HOLD_FINAL_SOURCE_PYTHON_CUPY_CAPABILITY_PAYLOAD_RD_AND_HBM
```

The review confirmed the fail-closed capability boundary, heterogeneous legacy
dependency-root rules, fixed source aperture, literal rank-prefix replay,
role-stratified gate, exact physical accounting, decoded-byte rescore, and
source-first control ordering. It did not run Python, initialize CuPy, read a
model payload, certify a deployment bootstrap, measure HBM traffic, or produce
an RD result.

Seventeen source-only tests passed on a pre-hardening source revision. Because
the final repaired source could not be rerun without a working Python/RunPod
environment, that earlier result is recorded but is not promoted to the final
closure. Runtime status is therefore honestly `HOLD`, not `PASS`.

## WFA evidence boundary

The `0.1675415` bits/symbol sparse unifilar-WFA saving was obtained only on a
source-free parity fixture. It established that the implementation could see a
long-range suffix-invisible dependency of the necessary scale; it did not show
that Qwen contains such a dependency.

The available raw-Qwen fixed-label producer result is adverse:

```text
pooled held-out saving                  -0.004629629629629629 bpw
raw payload minus full model            -0.0019661232277199073 bpw
physical rate                            2.5127314814814814 bpw
physical saving                         -0.01273148148148148 bpw
relative MSE                             0.030902167403153148
F                                        1.0064774170576134
fold savings                  -0.00694444, -0.00347222, -0.00347222
producer disposition          HARD_KILL_PRIMARY_PHYSICAL_RATE_OR_F
```

An independent WFA result audit was running on the RunPod when the SSH endpoint
became unavailable. No independent `PASS` receipt was recovered. On reconnect,
the first action is to inspect the old process, log, and exit receipt without
overwriting them; an absent result must be reported as absent rather than
reconstructed from the producer claim.

## Current branch ledger

- Fixed graph/Krylov residual refinement is closed: its nominal continuous
  gain was reproduced by the matched geometry control, leaving only
  `0.00015723251757482348 bpw` source-specific advantage.
- Fixed-rank TACTIC-DH384 is closed: capture `0.0936186` is far below the
  required `0.3238702221`.
- Global RM row-order and BMP/QTT/ROBDD packages are source/audit artifacts,
  not Qwen RD results. Their payload launch remains held.
- GF(2) recurrence remains a bounded source gate. Its raw selected-decision
  floor is adverse; only exceptional very-low-complexity structure remains
  plausible and its final tenth test awaits runtime recovery.
- Clustered same-layer CBIB-1 has now completed on the authenticated Qwen layer-15
  panel on the local RTX 3060.  The child emitted a complete result and exited
  successfully; the outer capability wrapper subsequently labelled the attempt
  failed only because CuPy wrote its harmless `CUDA path could not be detected`
  warning to stderr.  A detached source-only audit authenticated the six
  execution files, result equations, controls, and read fractions and sealed
  the completed child result at the manifest and root above.

  The scientific result is an adverse one.  Pair groups showed a misleadingly
  large `0.45979510004777663 bpw` gross private-stream saving, but the common
  stream cancelled it: the ideal net gain was only
  `0.000010730760043135371 bpw`, and the fully charged gain was
  `-0.00357806490188306 bpw`.  Its maximum matched-control charged gain was
  `-0.0035952639552799446 bpw`; the control-corrected source gain therefore
  remained `-0.00357806490188306 bpw`, versus the required
  `0.22933495044437174 bpw` on Up plus Down when Gate is unchanged.  Groups 4,
  8, and 16 were likewise net-negative after costs.  All ideal-capacity page
  layouts stayed below the read cap (maximums from `1.4193741857450828x` to
  `1.9294613996549124x` across the two rate endpoints), but they are explicitly
  not emitted finite codecs.  Thus same-layer fixed-label conditional entropy is
  killed by absent net source advantage, not by routed-read amplification.
- Ramanujan-384 remains scientifically open, but its r3 deployment is blocked
  at the execution boundary and has no staging or payload authority.
- Flexible-label PAIRPATH remains conceptually important because it searches
  nearby label paths jointly with conditional codelength, so a negative
  frozen-label CBIB census will not close it.  Its r1 source package passes its
  own full self-test at manifest
  `38eb78583c0013bfce9c1aaeace8e706bc455411daa43d68a1019cb5b041e3a1`,
  but an independent hostile audit correctly blocks payload work.  Only the
  complexity-asymptotic defect is fully closed; the executable aperture,
  literal packet/decoder, and complete control/confidence refit pipeline remain
  absent.  The sealed BLOCK audit has manifest
  `ab926aeb2bc610374871b70c635ea4720f6871b22ccf04f62f694e2f1d732ca3`
  and source root
  `6662dff728f399bfafbde664d1017a2d4efb4139a18cb493636c024d5a6b9d63`.

No gains from these separately fitted branches are additive. Promotion
requires one reconstructed physical object and one rate/MSE/read ledger.

## Continuation order after local runtime recovery

1. Seal the copied CBIB-1 r3 source-free RunPod evidence and publish its exact
   external hashes.  This is complete.
2. Preserve the completed local RTX-3060 CBIB child result and independently
   audit it.  Do not spend the still-unused remote authority on a duplicate
   payload run unless the local evidence fails audit.
3. Record the fixed-label CBIB hard kill after audit, including the distinction
   between gross private saving and near-zero joint/common-plus-private saving.
4. Replace the blocked PAIRPATH design shell with an actually integrated tree
   learner, latent fitter, flexible-label selector, finite packet decoder and
   full-refit controls before requesting any payload authority.
5. Run the bounded flexible-label pair experiment first and hard-kill it early
   unless joint label movement produces a source-specific rate-distortion gain.
6. Promote only a survivor into one literal 2.15--2.5-bpw object, then
   independently decode and rescore MSE, `F`, and routed reads.
7. Return to Ramanujan only after repairing and independently testing its
   symlink resolution and pre-validation one-use boundary.

The local RTX 3060 path is operational and produced the CBIB Qwen result above.
The most recently supplied RunPod mapping at `74.2.96.53:13725` subsequently
returned to a TCP-closed state; the original port `12079` was also closed.  Its
sealed one-use Qwen authority remains unused.  This does not block source-only
development, local audits, or bounded local RTX-3060 payload work.
