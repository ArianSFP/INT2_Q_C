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
- Ramanujan-384 is the next payload experiment because non-dyadic periods are
  not covered by the completed `{1,2,4}` cyclostationary screen and its early
  kill costs only 48 source-fixed blocks.

No gains from these separately fitted branches are additive. Promotion
requires one reconstructed physical object and one rate/MSE/read ledger.

## Resume order after runtime recovery

1. Authenticate and inspect the unfinished WFA audit files and old PID; do not
   overwrite them.
2. Execute the final frozen Ramanujan source tests and independent verifier.
3. Run the atomic-v3 CuPy runtime audit under an independently pinned worker.
4. Seal a non-self-authored external capability against the exact closures.
5. Run only the 48-block Ramanujan aperture.
6. On a miss, record the hard kill and stop the branch before controls.
7. On a survivor, emit the literal 2.4930556-bpw packet, decode/rescore it,
   then run the phase and eight Gaussian controls.
8. Freeze an independent result audit before making any performance claim.

At the time of this checkpoint the supplied RunPod mapping accepted a TCP
connection intermittently but refused or timed out before an SSH banner. No
remote command or payload access occurred during those retries.
