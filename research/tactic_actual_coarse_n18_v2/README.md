# UNIPOLAR-N18-307 v2 source closure

Status: **postimplementation source-only candidate; runtime deliberately
blocked**. This sibling does not modify or inherit the authority of
`tactic_actual_coarse_n18_v1`. It has no payload, no lower-rate artifact, no
Qwen measurement, no numeric import, and no CUDA execution.

## Why this package exists

The frozen TACTIC-DH384 experiment needs a genuine coarse reconstruction at
`307/128 = 2.3984375` physical bits per weight before its 384 fine bits can be
evaluated honestly. V1 described that stream but left important source,
portability, canonicality, path and publication boundaries open. V2 closes
those *source* boundaries without pretending that the numerical producer or
runtime has been authenticated.

This format is for arbitrary positive SwiGLU-MoE expert shapes:

```text
Gate: [intermediate, hidden]
Up:   [intermediate, hidden]
Down: [hidden, intermediate]
```

Down is transposed into `[intermediate, hidden]` canonical coordinates on
ingest. Model, checkpoint, vendor, layer, expert identity, provenance,
activations and external reference weights never select a seed, profile,
transform or decoder table. Qwen's `768 x 2048` triplets are one evaluation
geometry, not decoder side information.

## Exact universal packet

Each `N=2^18=262,144` canonical tile occupies one fixed reservoir:

```text
128-byte version-2 header
78,464-byte arithmetic reservoir including terminal zero fill
--------------------------------------------------------------
78,592 bytes = 628,736 physical bits = 307/128 bpw on a full tile
```

The test-channel profile remains `1.75 + 164/256 = 2.390625 bpw`, or
626,688 nominal logical bits. V2 has a 1,024-bit logical reserve. Overflow is
a terminal failure: no seed, profile, transform, topology or retry may change.

The header binds canonical role, rows, columns, tile ordinal, valid values,
shape/role/coordinate-derived SC and RHT seeds, the transmitted FP32 scale,
logical length, payload SHA-256, algorithm identifier and header CRC32. The
expert ordinal is not a seed input.

Logical arithmetic payloads are MSB-first with a hard logical EOF. Zero
extension is forbidden by the v2 contract. Unused low bits of the terminal
byte and every unused reservoir byte are zero. A future independent arithmetic
decoder must regenerate every causal frequency and decision, then perform a
byte-identical canonical decode/re-encode. `n18_common.py` independently tests
the envelope bit language; it is not a substitute for the future arithmetic
round trip.

For a partial final tile, the shape-bound suffix is implicitly zero padded,
but the complete 78,592-byte reservoir is still physical and charged. Thus
non-divisible shapes can have a rate above `307/128`; V2 never hides tail cost.

## Exact Qwen evaluation ledger

For the existing six equal-shape evaluation triplets:

```text
6 streams / matrix
18 streams / expert triplet
108 streams / panel
1,414,656 coarse bytes / triplet
8,487,936 coarse bytes / panel
67,903,488 / 28,311,552 = 307/128 bpw
```

Every selected expert owns and reads its 18 contiguous reservoirs once, so
the coarse-only equal-geometry read amplification is exactly `1x`. The common
ledger also handles unequal shapes using exact owner-aware rational
numerators and denominators; it does not divide every expert by an equal byte
share when their physical geometries differ.

## Security and closure boundary

Future runtime input uses canonical absolute paths. Every directory component
and leaf is opened without following symlinks, the regular file is retained by
a held descriptor, size and SHA-256 are checked, and identity/content are
revalidated before completion. Counts and checked products are bounded before
any source-proportional open, loop or allocation.

Output uses a private flat staging directory, `O_EXCL` and `O_NOFOLLOW`, file
and directory `fsync`, an artifact index, and `COMPLETE.json` written last.
The staging directory becomes visible only by
`renameat2(RENAME_NOREPLACE)`, followed by a parent-directory `fsync`. There is
no partial final output and no overwrite fallback. These held descriptor and
completion-last primitives are deliberately POSIX/RunPod-only.

`preflight_gate.py` has a second auth-before-import bootstrap. It requires the
exact manifest SHA-256 supplied externally, authenticates every source member,
checks the literal action token, reserves output before opening review or
runtime inputs, and requires a manifest-bound independent review receipt.
Only then may it inspect the environment lock, source plan or dependency graph.
It never imports NumPy, SciPy, CuPy or NVML and never launches CUDA.

## Dependency and telemetry boundary

`dependency_graph.json` pins the exact prototype polar encoder and independent
decoder Python sources by byte count, SHA-256 and complete AST import-root set.
A future POSIX preflight reads them through held descriptors and copies their
authenticated bytes to a private import snapshot.

That is not enough to authenticate native numerical code. The runtime
environment is deliberately unfrozen in `runtime_environment_lock.json`, so
every preflight currently fails before the source plan, weight payload,
numeric import or CUDA context. A replacement lock must bind the interpreter
and complete file roots for NumPy, CuPy, SciPy and `nvidia-ml-py`; because that
changes a manifest member, it requires another independent review.

The future CuPy receipt must include:

- exact logical bytes of explicitly enumerated H2D, D2H and model H2D arrays
  (not a claim about physical PCIe traffic);
- CUDA-event H2D, kernel and D2H intervals plus wall time;
- host RSS, NVML current-process/device VRAM and CuPy-pool baseline, peak and
  baseline-subtracted delta;
- interpreter/distribution versions, driver/runtime and compute capability;
  and
- CUDA-logical-to-NVML-physical UUID and PCI equality.

The pinned prototype encoder is not yet instrumented to supply that complete
receipt, and this package intentionally contains no numeric producer bridge.
Those are honest remaining dependencies, not inferred passes.

## Source before controls

The only authorized scientific order after a future independent review is:

1. authenticate all source/runtime inputs;
2. evaluate the absolute three-role source pilot for frozen DH384;
3. stop DH384 full-panel production if every role fails its predeclared
   absolute threshold;
4. only after a source survivor, generate decoded-Gaussian and
   structure-destroyed controls; and
5. never subtract a control result from the absolute source gate.

The source-leaking best-of-64, per-4,096-weight selector pilot dominates the
single frozen DH384 selector only. Its failure can stop DH384. It cannot kill
CAGE: it does not dominate coarse-programmed graph lifting, posterior
centroids, adaptive refinement trees, syndrome/coset decoding, or other CAGE
mechanisms. No result from this package is a converse for those branches.

## Verification

After the manifest is sealed, the source-only verifier and hostile suite are:

```bash
/usr/bin/python3.12 -I -B verify_source.py --package /absolute/package --repo-root /absolute/repository
/usr/bin/python3.12 -I -B /absolute/package/test_source_only.py -v
```

They use the standard library only. POSIX-only hostile cases cover held-input
leaf/ancestor symlinks, dependency snapshots, completion-last publication and
no-replace behavior. The checked-in environment lock must fail closed; passing
the source verifier does not authorize its replacement.

## Claim and authorization boundary

This package proves only a closed, universal packet/shape/rate/read design and
the source/runtime gates around a future producer. The three action strings
are necessary but never sufficient. A manifest-bound independent review and a
new frozen runtime lock are also mandatory. This source package authorizes no
payload and no CUDA, and it claims no lower-rate artifact, Qwen MSE, finite
TACTIC result, CAGE gain or universal performance.
