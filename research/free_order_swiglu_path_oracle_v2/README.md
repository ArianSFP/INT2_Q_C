# FOSP-ARX v2-DIRECT: charged free-order SwiGLU path gate

## Source-only verdict

This is a distinct successor to the immutable, independently blocked
`free_order_swiglu_path_oracle_v1`. It repairs the v1 scientific containment
error by deleting the three separate-role marginal-KLT early kill. The first
scientific gate in v2 is the full cross-role 3-by-3 predecessor family itself.

No Qwen/model matrix, pinned-panel datum, validation datum, CuPy module, CUDA
API, or GPU was accessed while producing this package. The package is
**deployment-blocked**. It includes code for a source-free CuPy calibration
and an auxiliary source run, but neither is evidence that those executions
occurred and neither is authorized by this source closure.

A cheap auxiliary execution is scientifically justified **only after** a new
independent source audit, a source-free full-geometry CuPy calibration, an
independent audit of that runtime receipt, and an external one-shot
authorization. The reason is narrow: the direct family was never measured;
the v1 failure invalidated its routing gate rather than measuring or killing
the family. Two fixed non-pinned auxiliary experts and a deterministic gross
upper-bound early stop keep the experiment bounded. A survivor would justify
only a finite residual-codec experiment, never a pinned-panel claim.

## The architecture and its unavoidable rate

Put a Qwen SwiGLU expert in common neuron orientation:

```text
G = gate_proj       in R^(768 x 2048)
U = up_proj         in R^(768 x 2048)
D = down_proj.T     in R^(768 x 2048)
```

For one permutation matrix `P`, deploy

```text
G' = P G,       U' = P U,       down_proj' = down_proj P^T.
```

Elementwise SiLU and multiplication commute with the common permutation, so
this preserves the expert function. An encoder can choose a path through the
768 joint `[Gate, Up, Down.T]` neurons and encode each neuron from its
predecessor.

The function symmetry is not free under the frozen score. The score compares
an independently decoded tensor with the original BF16 arrays in their
original coordinates. A canonical orbit representative does not reveal the
original arbitrary labels. Therefore the eligible frame serializes the
factoradic rank of the chosen permutation and scatters all three reconstructed
roles back through its inverse before MSE scoring.

```text
ceil(log2(768!))       = 6,260 information bits
physical fixed field  = 783 bytes = 6,264 bits
```

The four high padding bits are charged. Hiding order in codeword choices also
consumes rate and is forbidden as an uncharged workaround.

## What changed from v1

V1 used three independent 768-axis KLT spectra, one per role, as a supposedly
favorable early envelope. Its later stage allowed a full 3-by-3 cross-role
regression. The former does not contain the latter because it omits every
cross-neuron, cross-role covariance.

The independent audit supplied an exact counterexample. Choose 768
orthonormal zero-mean vectors `e_i` in the 2,047-dimensional zero-mean
subspace. Give neuron `i` the roles `(e_i, e_(i+1), e_(i+2))`, modulo 768.
Every separate-role neuron covariance is flat, so v1 stage 0 reports `s=0`.
The legal path `0,1,...,767` nevertheless predicts two of three role vectors
on every edge:

```text
residual ratio = 770 / 2304 = 0.3342013888888889
s              = 0.7906051829300244 bpw
```

That exceeds the fully side-adjusted v2 threshold
`0.1858070514584381 bpw`. V2 cannot kill this example before the direct stage
because it has no marginal stage. The independent v2 audit is required to
replay this exact routing property.

The alternative audit suggestion—one free joint 2,304-axis KLT—was not used.
It is expensive, and translating an ideal continuous joint transform curve
into a rigorous upper bound on a source-selected, nonorthogonal predictive
transform requires additional assumptions. Removing the prefilter is simpler
and logically exact. V2 grants no KLT, Jacobian, learned transform, or dense
source-specific map any scientific credit.

## Direct cross-role gate

For target joint neuron `Y` and distinct predecessor `X`, each `3 x 2048`, the
source oracle fits exactly nine coefficients:

```text
A*      = (Y X^T) (X X^T)^-1
capture = tr((Y X^T) (X X^T)^-1 (X Y^T)).
```

There is no intercept and no coordinate-wise parameter. Cross-products,
three-by-three inverses, captures, energies, residuals, and acceptance
statistics use CuPy float64. The 768-by-768 capture matrix alone is copied to
host for deterministic SciPy assignment.

The stage reports three nested objects:

1. **Containing relaxed upper bound.** Every one of all 768 targets chooses
   its best non-self predecessor independently. Reuse, cycles, and the missing
   anchor are allowed. Any legal 767-edge path uses a subset of target-wise
   choices and pays an uncaptured anchor, so its exact linear-regression
   capture cannot exceed this sum.
2. **Achievable exact path.** A maximum-weight non-self cycle cover is cut at
   the weakest edge of every cycle. Segments are ordered deterministically and
   explicit bridges produce one permutation. This is achievable but not
   claimed optimal.
3. **Achievable FP16 replay.** Exact coefficients select the path. All nine
   coefficients on every selected edge are then rounded through IEEE binary16
   and residual energy is recomputed directly. The exact 13,806 coefficient
   bytes are hashed in edge order.

This separation matters. Failure of the relaxed upper bound is a valid hard
kill for this frozen linear path family. Failure of the cycle-cover heuristic
while the relaxation survives is only an ambiguity and routes to a stronger
path solver. Success of the FP16 replay is only an opportunity survivor because
the residual payload has not been quantized or serialized.

### Identically processed Gaussian controls

Eight fixed controls are constructed separately for every expert and every
neuron. Each control preserves:

- all three role means across the 2,048 model coordinates;
- the exact centered `3 x 3` Gate/Up/Down Gram;
- every role energy, cross-role covariance, and heteroskedastic scale implied
  by those moments.

Only centered orientation is randomized. Each control receives the identical
all-pairs search, reuse relaxation, cycle cover, path construction, exact fit,
FP16 rounding, and residual replay. No source-only KLT or Jacobian is fitted
and then reused on easier controls.

For each of the relaxed, legal-exact, and legal-FP16 metrics, v2 computes

```text
U = s_Qwen - mean(s_control)
    + 3*hypot(SE_control_Monte_Carlo, SE_delete_one_expert).
```

Only Qwen-specific excess over identically optimized controls receives credit.
The `+3 SE` value is an optimistic upper confidence statistic suitable for a
fail-safe early kill. It is not a lower confidence proof of improvement.

### Frozen early stops

All physical side bits increase the required gross structural gain from

```text
-0.5*log2(0.8) = 0.16096404744368115 bpw
```

to

```text
0.16096404744368115 + 117224/4718592
= 0.1858070514584381 bpw.
```

The runner stops according to these exact rules:

- If the gross Qwen relaxed exact `s` is below `0.1858070514584381`, stop
  before generating controls. No legal charged path can reach the target.
- Otherwise run all eight matched controls. If their corrected relaxed upper
  statistic is below the same threshold, hard-kill the frozen linear path
  family.
- If the relaxation survives but the legal FP16 path misses, report an
  optimization-gap ambiguity. Do not call it a family kill.
- If gross and corrected legal FP16 results survive, stop with a source-oracle
  survivor and require a separately frozen residual codec.

## Exact physical and read ledger

One expert has `4,718,592` weights. The only eligible bridge freezes:

| Field | Bytes | Bits | bpw |
|---|---:|---:|---:|
| Header | 64 | 512 | 0.0001085069 |
| Factoradic permutation | 783 | 6,264 | 0.0013275146 |
| `767 * 9` FP16 coefficients | 13,806 | 110,448 | 0.0234069824 |
| **Total side** | **14,653** | **117,224** | **0.0248430040** |

The frame has `floor(4,718,592*R/8)` bytes. Side fields reduce the residual
payload reservoir; they never expand the cap.

| Requested R | Frame bytes | Actual bpw | Residual-payload bpw | Pessimistic cold read |
|---:|---:|---:|---:|---:|
| 2.15 | 1,268,121 | 2.1499989827 | 2.1251559787 | 1.0045224391x |
| 2.30 | 1,356,595 | 2.2999996609 | 2.2751566569 | 1.0054349308x |
| 2.50 | 1,474,560 | 2.5000000000 | 2.4751569960 | 1.0027777778x |

Logical compressed-object traffic is exactly one expert frame. Cold-page
accounting rounds that frame to 4-KiB pages and pessimistically rereads one
shared-manifest page. It remains below `1.006x`, far under the strict `2x`
limit. Decoder scratch keeps one previous joint BF16 neuron (`12,288` bytes),
but scratch is not compressed-object read traffic.

## Three firewalls

### Source firewall

`source_bindings.json` fixes two non-pinned auxiliary experts and all six
matrix hashes. The production runner has one source selector:
`--workspace-root`. It has no alternate manifest, individual matrix, pinned
panel, validation, or target argument.

The root directory descriptor is opened before heavy imports and held for the
entire run. Each fixed relative path is traversed descriptor-relative with
`openat`/`O_NOFOLLOW`; every directory component and final file must be real,
and the exact bytes hashed from the held descriptor are the bytes decoded.

### Runtime/evidence firewall

An authorization hash and PASS-looking strings are insufficient. Before CuPy
import, the runner itself opens all five canonical absolute evidence paths
named by the authorization:

- source-audit `AUDIT_SHA256SUMS.txt` and `audit_receipt.json`;
- source-free `runtime_receipt.json`;
- runtime-audit `AUDIT_SHA256SUMS.txt` and `audit_receipt.json`.

It checks both audit manifests bind their receipts and independent verifiers,
then verifies literal schemas and statuses, canonical internal seals, the
exact package and runtime targets, the v1 counterexample replay, and every
zero-access field. The runtime audit must bind both the file and internal hash
of the source-free receipt. The authorization must bind the bytes the runner
actually opened. Its runtime tuple must exactly equal that opened receipt and
the observed Python/NumPy/CuPy/SciPy/GPU/CUDA tuple.

The authorization builder is convenience code, not a trust root. The runner
repeats all material validation independently.

### Output firewall

Package, source, output, authorization, source-audit, runtime-receipt, and
runtime-audit roots must be pairwise disjoint. CLI and authorization paths
must use one normalized, canonical, absolute spelling with no symlinked
component. The output must not exist.

The runner holds the output-parent descriptor from preflight through the GPU
work, rechecks its device/inode identity, and commits with descriptor-relative
`O_CREAT|O_EXCL`. Replacing the pathname of that parent cannot redirect the
write. The frozen package is never an output location.

## Required external receipt contracts

The independent source audit receipt must use schema
`free-order-swiglu-path-v2-independent-source-audit-receipt-v1`, literal status
`PASS_V2_INDEPENDENT_SOURCE_AUDIT`, immutable PASS artifact status, a canonical
unsigned SHA-256, the exact five-field `audited_package` object required by the
runner, a `PASS_COUNTEREXAMPLE_REACHES_DIRECT_STAGE` replay, and zero model,
pinned, validation, CuPy, CUDA, GPU, and external-fetch access.

The runtime audit receipt must use schema
`free-order-swiglu-path-v2-independent-runtime-audit-receipt-v1`, literal
status `PASS_V2_INDEPENDENT_RUNTIME_AUDIT`, immutable PASS artifact status, a
canonical unsigned SHA-256, the same exact package target, exact file/internal
runtime-receipt hashes, and zero model, pinned, validation, production result,
production GPU, CuPy-import, CUDA-API, and GPU-device access by the auditor.

Both audit manifests must include `audit_receipt.json` and
`verify_audit.py`. These exact contracts are enforced by both
`create_authorization.py` and `free_order_oracle_v2.py`.

## Reproduction routing

Pure-standard-library source checks import no CuPy:

```bash
python -B test_source_only.py
python -B verify_package.py
```

After a new independent source audit, copy the exact sealed package to a
source-disjoint staging directory. Run the source-free calibration only into
a new, disjoint evidence directory:

```bash
CUDA_VISIBLE_DEVICES=0 /audited/venv/bin/python -B \
  /sealed/fosp-v2/calibrate_runtime.py \
  --output /new/runtime-evidence/runtime_receipt.json
```

After an independent runtime audit, an external controller may create one
authorization. All shown paths must be canonical absolute paths and all
parents must already exist:

```bash
/audited/venv/bin/python -B /sealed/fosp-v2/create_authorization.py \
  --source-audit-manifest /sealed/source-audit/AUDIT_SHA256SUMS.txt \
  --source-audit-receipt /sealed/source-audit/audit_receipt.json \
  --runtime-receipt /sealed/runtime/runtime_receipt.json \
  --runtime-audit-manifest /sealed/runtime-audit/AUDIT_SHA256SUMS.txt \
  --runtime-audit-receipt /sealed/runtime-audit/audit_receipt.json \
  --workspace-root /workspace/INT2__compression \
  --output /new/production/result.json \
  --python-executable /audited/venv/bin/python \
  --authorization-output /new/authorization/authorization.json \
  --scope-literal FOSP_V2_AUXILIARY_DISCOVERY_ONLY_NO_PINNED_PANEL
```

Finally pass the printed authorization SHA-256 to the runner:

```bash
CUDA_VISIBLE_DEVICES=0 /audited/venv/bin/python -B \
  /sealed/fosp-v2/free_order_oracle_v2.py \
  --workspace-root /workspace/INT2__compression \
  --output /new/production/result.json \
  --authorization /new/authorization/authorization.json \
  --authorization-sha256 <exact printed SHA-256>
```

Even a successful auxiliary result must be independently audited before it
can influence a new architecture freeze. This package never authorizes a
pinned-panel run.
