# Lossy-tail peeling oracle v7

Status: **FROZEN SOURCE-ONLY CANDIDATE FOR A FRESH INDEPENDENT AUDIT. NO RUNTIME CALIBRATION, QWEN/PAYLOAD ACCESS, CUPY/CUDA/GPU EXECUTION, OR PRODUCTION AUTHORIZATION HAS OCCURRED.**

V7 is a distinct repair of preserved, blocked v6. It does not modify v6 or
the sealed v6 audit. It retains the same auxiliary six-expert/two-role cohort,
61-profile scalar-tail grid, physical rates 2.15/2.30/2.50 bpw, ideal-Gaussian
bulk oracle, four matched Gaussian controls, exact side-bit accounting, and
strict `<2x` logical and page-read limit. Its purpose is release correctness:
it closes the five blockers frozen by the independent v6 audit before any
runtime or payload work is considered.

This package is a producer candidate, not a source-audit PASS. A new independent
auditor must authenticate these exact bytes. The package contains no runtime
receipt, runtime-audit receipt, production authorization, or result.

## Five repaired blockers

### 1. A direct child cannot reuse preflight authority

`preflight_launch.py` is the sole authorized launcher. After it authenticates
the exact source audit, source-free runtime receipt and independent runtime
audit, filesystem/mount bindings, absent output, and full stage, it creates a
Unix `SOCK_SEQPACKET` socketpair and spawns the immutable stdlib bootstrap.

It sends exactly one canonical capability record and then EOF. The record
binds:

- live parent and child PIDs;
- `SO_PEERCRED` UID/GID/PID;
- a 256-bit random nonce;
- the exact live `/proc/<parent>/cmdline` hash;
- launch-manifest file hash;
- authorization file and internal hashes; and
- bootstrap and scientific-core hashes.

The child requires one record followed by EOF, recomputes all bindings,
acknowledges the canonical capability hash exactly once, closes the channel,
and only then descriptor-executes the authenticated core. Preflight validates
the acknowledgement and waits for that exact child. Missing capabilities,
regular-file descriptors, extra records, wrong peers/PIDs, altered parent
commands, and direct bootstrap entry all fail.

The core does not trust a supplied dictionary by itself. Before importing
NumPy it independently requires the exact production grammar, raw bootstrap
and core stage spellings, exact stage arguments, live parent/child PIDs, the
preflight executable inode, the exact `/proc` parent command, and the bound
manifest/authorization hashes. `main()` reproves the same live parent.

### 2. Raw entry and stage authentication precede NumPy

`lossy_tail_oracle.py` is now a small stdlib-only production bootstrap.
`lossy_tail_core.py` begins with a stdlib production firewall and calls it
before linearly reaching `import numpy`. Direct core execution fails at the
first firewall. A forged production context whose parent is not the exact
preflight also fails under Python import tracing before NumPy appears.

The source-free runtime calibrator remains a separate stdlib entrypoint. It
validates raw `argv[0]`, exact stage closure/hashes, repair/manifest seals,
explicit `CUDA_VISIBLE_DEVICES=0`, and a disjoint create-new output before it
descriptor-executes the core in runtime-calibration mode and imports CuPy.

### 3. Read-valid candidates cannot be erased

Every uniform raw profile and every uniform support-XKLT profile is evaluated.
At each coordinate-descent position every eligible profile participates in
the trial list. Each scientifically valid scored row first passes finite and
read-ledger validation. Ranking is then performed *inside* rows whose maximum
cold logical and maximum cold page amplification are both strictly below
`2.0` for every expert.

Only read-valid winners seed coordinate descent, only read-valid trial winners
advance, and final allocation-bearing rows are rescored and revalidated. An
unconstrained winner is retained solely in a clearly named diagnostic field;
it can never become a calibration or decision row. Per-rate/mode ledgers record
uniform, XKLT, and coordinate trial counts and executable all-profile coverage
booleans.

The adversarial CPU panel deliberately gives the global `F=0.1` winner a
`2.25x` read cost and a worse `F=0.2` row a `1.25x` read cost. Raw-uniform,
raw-adaptive, and support-XKLT all retain the `1.25x` row while exposing the
invalid global row only as a diagnostic.

### 4. Pool release follows every live consumer

For each of 48 source-free runtime cells, both FP32-affine sentinel gathers
are hashed and reduced before release. The exact live arrays
`raw/bits/rounding/words/zbf/table_words/table/affine_table/gathered/rng` and
host hash buffers are then deleted, the active stream is synchronized, and
the default pool must report `used_bytes()==0` *before* `free_all_blocks()`.
After free, both used and total bytes must be zero. Equivalent closure is
required for all five stable-order adversaries.

Every cell records the before/after values and deletion assertion. CPU AST
mutation tests remove `affine_table`, `gathered`, and `rounding` individually
from the barrier and prove each mutation is rejected.

### 5. Nonfinite values always fail closed

All valid scored rows validate rate, distortion, `F`, `s`, source energy,
side bits, maximum read values, and every expert amplification before ranking.
Calibration validates every Qwen/control row, each individual score, the
four-control mean and sample standard deviation, excess, calibrated `F`, and
required-score fraction. The decision recursively rejects every nonfinite
numeric leaf before selection, then revalidates absolute, calibrated, joint,
selection, boundary, and distance scalars.

`NaN`, `+inf`, and `-inf` injected into individual Qwen scores, calibrated
scores, nested control scores, aggregate means, rates, and scored candidates
all raise. Positive infinity cannot win a `max()` promotion.

## Frozen decision contract

Let `s_abs=-0.5*log2(F_Qwen)` and
`s_cal=s_Qwen-mean(s_control)`. A row's joint score is
`min(s_abs,s_cal)`.

- `s* = 0.16096404744368115` (`F=0.8`).
- The optimistic envelope is the maximum joint score over read-valid
  `free_lloyd` and `zero_tail_error` rows.
- `EARLY_KILL_FAR_SHORT` requires the optimistic score to be below
  `0.14096404744368115`.
- Values from that guard through `s*` hold rather than kill.
- A finite FP16 row warrants later codec work only if both its absolute and
  calibrated scores reach `s*` and the optimistic envelope is consistent.
- Any decision value within or equal to `1e-4 bpw` of either threshold becomes
  `HOLD_NUMERIC_BOUNDARY`; it can neither kill nor promote.

No tail gain is added to an earlier architecture. Every row recomputes one
same-rate tail-plus-ideal-bulk allocation with complete bit closure.

## Source-free runtime contract remains unexecuted

A future, separately authorized runtime calibration must create 48 complete
vectors: four replicas by twelve matrices. It freezes raw float32 and RNE-BF16
hashes, two FP32-affine gather hashes per cell, float64 mean/variance hex
values, five stable-order adversaries, memory-closure evidence, the exact
runtime tuple, and an aggregate hash. It accepts no source/model argument.

That receipt is explicitly untrusted. A separate independent audit must seal
it before a later one-shot production authorization can bind it. Production
must replay the entire receipt before opening the first source file.

## Exact source-stage verification

`launch_manifest.json.allowed_members` defines an exact eleven-file stage.
Copy only those members into a new isolated directory; do not copy this README,
tests, receipts, artifact manifest, or producer verifier into the stage.

From any working directory:

```bash
python3 -B -I /ABS/STAGE/audit_lock_entrypoint.py \
  --manifest /ABS/STAGE/launch_manifest.json \
  --manifest-sha256 3d5bc5ed95071cc45406d0d2906b54f40d32adad0dffc6323b8fa80ca491ed63
```

The audit entrypoint rejects `-O`, relative/dot/symlinked raw launchers,
duplicate JSON keys or rows, any added file/directory/device/link, omitted
members, byte/hash drift, and repair identity drift. It does not accept a
source, runtime receipt, authorization, or output parameter.

Producer-only checks (not independent audit authority):

```bash
/workspace/int2-cupy-venv/bin/python -B -I test_lossy_tail_core.py
/workspace/int2-cupy-venv/bin/python -B -I test_release_security.py
python3 -B verify_package.py
```

The two suites run 28 source/CPU tests. The production-shaped test uses an
empty synthetic source directory, fake source-free external audit receipts,
and a tiny authenticated core stub. It reaches one capability-bound boundary
with NumPy and CuPy absent, opens no payload, and creates no production output.

## Release sequence

1. Freshly and independently audit these exact source bytes. No payload,
   model path, CuPy, CUDA, GPU, runtime receipt, authorization, or result may be
   touched.
2. Only after source PASS and separate approval, run the source-free runtime
   calibrator once under the frozen runtime.
3. Independently audit and seal that runtime receipt.
4. Only after another separate approval, create a one-use production
   authorization binding all source/runtime/audit/filesystem identities and an
   absent output.

No step in this producer package grants authority for a later step.

## Claim boundary

The scientific scheme is still a favorable auxiliary oracle: sparse scalar
tail values plus an ideal Gaussian bulk channel. It is not a finite end-to-end
codec and not a converse for learned masks, vector codebooks, semantic expert
structure, Gate matrices, other layers, or arbitrary block families. A
producer test PASS, independent source PASS, runtime PASS, optimistic survivor,
or numeric hold is not a Qwen compression result.
