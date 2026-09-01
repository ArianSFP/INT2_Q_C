# Independent result audit: MALT64 decoded-SVD tangent stage 0

Verdict: **PASS**, scoped to the exact frozen MALT64-r3 continuous-tangent
stage-0 screen. This is not a finite codec, a lower-coarse-rate measurement,
fresh-validation evidence, or a converse for other decoder-conditioned
families.

## Evidence and replay

The audit binds the exact three-file producer closure, the independently
published plan and header, the recorded decoded post-KLT hash, all 18 plan
source identities/hashes, and a new disjoint RunPod result. The replay used a
byte-identical producer script, an absent output path, batch size 32, and only
the already-open plan inputs. Its output has SHA-256
`9de04f91831c7da04f1b908d8cd6381aeaf263dfd0a7e1e7556934e214ade1a5`
and is structurally identical to the producer result after removing only
`execution.elapsed_seconds`. All 18 matrix values, 6 expert folds, 3 role
folds, jackknife values, bindings, and the decision reproduced exactly.

The supplied `independent_gpu_replay.py` expresses the projection energy by
the independent identity

```text
||U^T E||_F^2 + ||E V||_F^2 - ||U^T E V||_F^2,
```

rather than constructing the producer's `P_U E + E P_V - P_U E P_V` matrix.
It holds and hashes the plan, header, 226,492,416-byte decoded artifact, and
each of the 18 sources while scoring. Upload of this locally authored script
to the external endpoint was denied by the sandbox, so this second full GPU
implementation was not executed. The verifier records this limitation and
tests the two formulas independently on 16 deterministic synthetic problems.

## Independent reconstruction

The plan is exactly 24,790 bytes with SHA-256
`8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868`
and valid internal canonical seal
`99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`.
The 128-byte header hashes to
`3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538`;
all six decoded rotation pairs are finite and nondegenerate. The decoded
artifact binding is
`af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0`.
The plan contains exactly 18 ordered Gate/Up/Down sources with safe relative
paths, expected geometries, three matrices per expert, and no path containing
`validation`.

The rank-three tangent dimension is independently derived as
`3*(64+64-3)=375`, giving null share `375/4096=0.091552734375`. The result
scores all `18*384=6,912` blocks. Re-aggregation of the 18 matrix rows gives:

- source energy: `16192.894508855934`;
- coarse SSE: `500.39553685426534`;
- tangent projection energy: `45.899290453794244`;
- capture: `0.09172601886567566`;
- delete-expert three-SE upper bound: `0.09185862912923075`.

Every matrix projection energy lies between zero and its error energy. All
matrix ratios, expert/role sums, delete-one values, jackknife center/SE, and
aggregate fields were independently recomputed from result rows.

## Ledger, threshold, and decision

The exact ledger is `307/128` coarse + `384/4096` target + `1/128` metadata
= `320/128 = 2.5 bpw`; the cold-read factor is `73/72`. Under the explicitly
favorable transfer assumption,

```text
D0 = 0.9888693569009007 * 2^(-2 * 2.3984375)
   = 0.035574242296714034
target relative MSE = 0.8 * 2^(-2 * 2.5) = 0.025
required capture = 1 - 0.025/D0 = 0.2972443434920543.
```

The upper-three-SE capture reaches only `0.30903406958082696` of that
requirement. The runner hard-kills only from this frozen aggregate UCB, after
confirming exact base SSE/energy, so
`POLICY_REJECT_MALT64_R3_FAR_SHORT_STOP_BEFORE_CONTROLS` follows. Controls and
finite design are absent. This is an architecture-scoped policy rejection,
not a universal scientific converse.

## Verification

On this checkout, use the independently published checkpoint plan/header:

```powershell
python -B verify_audit.py `
  --audit-manifest-sha256 <EXTERNAL_PIN_FROM_RELEASE_HANDOFF> `
  --plan ..\..\results\qwen\strata_expert_affine_checkpoint\plan.lock.json `
  --header ..\..\results\qwen\strata_expert_affine_checkpoint\assets\header.bin
```

The verifier requires an external manifest pin, holds/hashes all locally
available evidence before use, strictly parses JSON, checks exact producer and
audit closures, and has no source-default or fresh-validation path. The
manifest pin is kept external because the manifest hashes the receipt.

