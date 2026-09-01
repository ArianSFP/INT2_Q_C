# Lossy-tail peeling oracle v8

Status: **FROZEN SOURCE-ONLY PRODUCER CANDIDATE FOR A FRESH INDEPENDENT AUDIT. NOT AUTHORIZED FOR RUNTIME CALIBRATION, QWEN/MODEL ACCESS, CUPY/CUDA/GPU EXECUTION, OR PRODUCTION.**

V8 is a distinct repair fork of the immutable v7 stage and the fresh
independent v7 BLOCK manifest whose exact file SHA-256 is
`120b616c726253a82850e93b720e48c56c2aa7af59f1c2b7ec288bec215e4621`.
Neither v7 nor its audit was modified. V8 preserves the v7 scientific
experiment: the auxiliary six-expert/two-role cohort, 61-profile tail grid,
2.15/2.30/2.50 physical bpw rates, ideal-Gaussian bulk oracle, four matched
controls, finite-only decision logic, numeric-boundary HOLD, strict early kill,
complete rate ledger, live support-XKLT angle charge, and strict `<2x` logical
and page-read rule.

This package contains source, contracts, producer tests, and integrity
manifests only. It contains no source-audit PASS, runtime receipt, runtime-audit
PASS, production authorization, Qwen result, or compression claim.

## The five v8 repairs

### 1. Immutable preflight provenance, not mutable cmdline text

The external entry to `preflight_launch.py` first authenticates its raw
absolute `argv[0]`, exact `__file__` identity, exact stage membership, and
manifested bytes. It then:

1. creates a Linux memfd with sealing enabled;
2. copies the exact manifested `preflight_launch.py` bytes into it;
3. applies `F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE`;
4. verifies type, size, device/inode, seals, bytes, and SHA-256; and
5. re-executes Python with `/proc/self/fd/<n>` as the script.

Only that sealed-descriptor execution may process production authority. The
one-record `SOCK_SEQPACKET` child capability carries the same memfd fd,
device/inode, byte count, seal mask, and SHA-256. The stdlib bootstrap and the
scientific core before NumPy independently compare the inherited sealed bytes
to the currently manifested stage bytes. The capability also binds
`SO_PEERCRED`, parent/child PIDs, a nonce, manifest and authorization hashes,
bootstrap/core hashes, and the held output-parent descriptor.

`/proc/<pid>/cmdline` is never read or trusted. `/proc` fd/executable identity
checks are supplemental live-process bindings; immutable sealed source bytes
are the provenance object. Direct bootstrap/core entry lacks the one-use
channel and inherited descriptors and fails closed.

### 2. PASS meaning is frozen; authorization cannot choose it

The authorization contract, preflight, and core contain the same literal
source-audit and runtime-audit manifest schemas, receipt schemas, and PASS
status strings. They also freeze the runtime calibrator receipt's pre-audit
status as `UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT`.

Authorization audit sections contain only paths plus exact file/internal
digests. `required_status` is not an accepted key. Because exact key sets are
enforced, adding it rejects; a later authorization cannot relabel a sealed
BLOCK receipt as acceptable evidence.

### 3. Output stays anchored to authenticated directory descriptors

During preflight, the existing output parent is checked against the authorized
path, mount row, device, and inode, then opened once with
`O_DIRECTORY|O_NOFOLLOW`. That exact descriptor and identity are passed to the
child and reauthenticated before NumPy.

After this gate the core never reopens an output absolute pathname. It checks
run-root absence relative to the held parent, creates the run root relative to
that parent, opens the new directory with `O_DIRECTORY|O_NOFOLLOW`, creates an
exclusive temporary result with `O_EXCL|O_NOFOLLOW`, writes and fsyncs it,
renames it to `result.json` with source and destination dirfds, fsyncs the run
directory, and finally fsyncs the held parent. Successful create-new output
continues to make authorization replay fail closed.

### 4. Complete six-field evidence for all 53 runtime rows

Each of the 48 RNG/BF16/FP32-affine cells and each of the five stable-order
adversaries emits exactly the complete memory evidence shape:

- `stream_synchronized`;
- `used_bytes_before_free`;
- `total_bytes_before_free`;
- `used_bytes_after_free`;
- `total_bytes_after_free`; and
- `all_per_cell_gpu_arrays_deleted_before_free`.

Every row synchronizes, requires zero used bytes before freeing cached blocks,
then requires both used and total bytes to be zero. The runtime contract and
source now agree exactly.

### 5. `build_panel` proves no live GPU loop aliases

Per expert, every affine/gathered value, histogram, mask, and paired-XKLT
consumer completes before release. The code explicitly deletes owning
containers and the otherwise-surviving loop aliases `masks`, `x`, and `words`,
synchronizes, asserts `used_bytes()==0`, frees cached blocks, and asserts
`used_bytes()==total_bytes()==0`. A per-expert ledger records the closure.

## Adversarial producer tests

`test_release_security.py` contains a mutation panel for every repaired
blocker. It rejects removal of memfd creation/sealing or either byte-equality
check; reintroduction of cmdline provenance; changed PASS constants or an
authorization-chosen status; loss of either output dirfd anchor, `O_EXCL`, or
parent fsync; removal of any required memory field; and retention of
`masks`/`x`/`words` or loss of synchronization/pre-free/post-free checks.

The inherited v7 CPU tests continue to cover:

- invalid global winners cannot erase a `<2x` eligible row;
- every uniform/XKLT/coordinate eligible row participates before ranking;
- live support-XKLT rank-one components still charge their angle;
- NaN, positive infinity, and negative infinity fail in individual and
  aggregate score trees;
- numerical-boundary values HOLD rather than kill or promote; and
- early-kill, target, rate, read, and decision equations remain frozen.

Tests contain no `assert` statements in release entrypoints, and production
rejects Python `-O`. The Linux production-shaped producer test uses only an
empty synthetic source directory and a tiny authenticated stdlib core stub; it
opens no payload and creates no result.

## Stage versus producer package

`launch_manifest.json.allowed_members` defines the exact eleven-file launch
stage. README, tests, producer receipt, artifact manifest, and verifier are
package evidence, not stage members. Copy only the eleven allowed members when
constructing an isolated stage. Added files, directories, links, devices,
duplicate rows, duplicate JSON keys, missing members, or byte drift reject.

Source-only stage replay, after substituting the final manifest digest printed
by `verify_package.py`:

```bash
python3 -B -I /ABS/STAGE/audit_lock_entrypoint.py \
  --manifest /ABS/STAGE/launch_manifest.json \
  --manifest-sha256 <V8_LAUNCH_MANIFEST_SHA256>
```

Producer-only checks (never independent audit authority):

```bash
/workspace/int2-cupy-venv/bin/python -B -I test_lossy_tail_core.py
/workspace/int2-cupy-venv/bin/python -B -I test_release_security.py
python3 -B verify_package.py
```

## Required release sequence

1. A fresh independent auditor authenticates these exact bytes, runs its own
   adversarial source review, and emits the frozen v8 source PASS schema/status.
2. Only after that PASS and separate explicit approval may one source-free
   runtime calibration run on the pinned runtime. It accepts no model/source
   path and emits an untrusted receipt outside the stage.
3. A different independent step audits and seals the exact runtime receipt.
4. Only a later one-shot authorization may bind exact source/runtime audit
   digests, source identity, interpreter, CUDA visibility, mount snapshot,
   filesystem identities, and absent descriptor-anchored output.

No earlier artifact authorizes a later phase. In particular, this producer
candidate must not be used to access Qwen, import CuPy, initialize CUDA, or run
the GPU experiment.

## Claim boundary

This remains a favorable bounded oracle: sparse scalar tails plus an ideal
Gaussian bulk channel on auxiliary matrices. It is not a finite end-to-end
codec or a converse for learned masks, vector codebooks, semantic expert
structure, Gate matrices, other layers, or arbitrary blocks. Producer PASS,
independent source PASS, runtime PASS, optimistic survival, and numeric HOLD
are not Qwen compression results or production authority.
