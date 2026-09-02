# UNIPOLAR-N18-307 v2 postimplementation review

Status: **implementation stopped for independent audit; no producer manifest,
payload authority or CUDA authority**.

This is the producer's final self-review boundary. It is not a
`SOURCE_MANIFEST.json`, independent audit, runtime lock, review receipt,
artifact receipt or scientific result. The producer intentionally did not
self-authorize a source manifest.

## Outcome

V2 is a clean format break in a new sibling directory. V1 and all audit
directories were left untouched. The implementation now provides:

1. a shape/role-portable N18 fixed-reservoir envelope for arbitrary positive
   SwiGLU-MoE Gate, Up and Down shapes, including explicit valid-tail binding
   and fully charged tail reservoirs;
2. a 128-byte v2 header with shape/role/coordinate-only seeds, transmitted
   FP32 scale, payload SHA-256, algorithm identifier and header CRC32;
3. an MSB-first hard logical EOF, zero terminal/fill rules and bounded
   byte-identical bit-language re-encode primitive;
4. exact `307/128` full-tile and Qwen-evaluation arithmetic plus owner-aware
   rational read accounting for unequal expert geometries;
5. a universal source-plan adapter that rejects model/checkpoint/layer/expert
   identity fields and validates all counts/products before payload opens;
6. an explicit source-first state machine in which controls cannot open until
   the absolute DH384 source pilot survives;
7. a frozen non-converse: failure of the best-of-64 rank-384 pilot can stop
   frozen DH384 only and cannot kill uncontained CAGE mechanisms;
8. component-wise absolute POSIX `openat`/`O_NOFOLLOW` inputs held by
   descriptor, streaming SHA-256, stable-identity revalidation and bounded
   metadata materialization;
9. private flat output staging, exclusive files, fsync, `COMPLETE.json` last,
   `renameat2(RENAME_NOREPLACE)`, parent fsync and complete public-tree rehash;
10. a preflight entry that authenticates the externally pinned complete source
    tree before sibling imports, reserves output before review/runtime inputs,
    and never imports a numeric library or launches CUDA; and
11. an exact held-FD/private-snapshot dependency graph for the two pinned
    prototype Python cores plus a mandatory CuPy/NVML telemetry schema.

## Defects found and repaired during self-review

The first implementation was not sealed. Review found and repaired:

- whole-matrix hashing materialized the source despite permitting large legal
  shapes; hashing and stability checks now stream in fixed chunks;
- pilot preflight opened every matrix declared in the plan; it now opens only
  the coordinate-selected first Gate/Up/Down triplet, while full mode alone
  may open all matrices;
- the publisher did not re-open and rehash the public directory after atomic
  rename; every artifact, index and completion file is now checked again;
- canonical re-encode first converted an arbitrary decision iterable to an
  unbounded list; it now enforces the physical bit cap while consuming;
- source stored-shape equality could admit Boolean dimensions by Python value
  equality; dimensions now require exact integer types;
- the independent prototype decoder's complete AST import graph initially
  omitted its nested CuPy imports; the corrected graph includes them; and
- source bootstrap rejected a symlinked package leaf but did not walk every
  ancestor by descriptor; it now opens every absolute package component with
  `O_NOFOLLOW`.

## Source-only execution evidence

The latest pre-review replay ran on the supplied RunPod with
`/usr/bin/python3.12 -I -B`. It used only standard-library code, synthetic
bytes, temporary files and the two already-pinned Python source files. It did
not import NumPy, SciPy, CuPy or NVML, initialize CUDA, enumerate model files,
or open any model payload.

The replay passed syntax compilation and 20/20 selected hostile tests in
0.377 seconds with zero failures, errors or skips:

- exact full/tail rate and owner ledgers, including unequal shapes;
- packet hard EOF, lengths 0/1/7/8/9/19/4095 and maximum physical capacity;
- header, digest, reserved, fill and repaired-CRC terminal-pad tampering;
- canonical decision mismatch and seed-domain separation;
- two different legal SwiGLU shapes and Down transpose semantics;
- forbidden identity fields, role disorder and hostile huge dimensions;
- source-before-control transitions and the DH384/CAGE non-converse;
- fail-closed unfrozen runtime and manifest/action-bound review receipts;
- complete held-FD/hash/AST authentication of both prototype dependencies;
- telemetry UUID/PCI/delta invariants;
- held-input leaf and ancestor symlinks; and
- completion-last publication and existing-target no-replace behavior.

The source-manifest closure/tamper tests and fail-before-source-plan subprocess
test are implemented but deliberately not run by the producer: they require a
`SOURCE_MANIFEST.json`, which must be authored and pinned by an independent
auditor rather than this producer.

## Explicit blocking dependencies

This source is **not ready for payload or CUDA**.

1. `runtime_environment_lock.json` deliberately has status
   `UNFROZEN_BLOCK_RUNTIME`; it contains no fake interpreter or distribution
   pins. `validate_environment_lock` fails before a source plan or payload can
   open.
2. The two pinned external files are prototype Python algorithm sources, not a
   complete authenticated native runtime.
3. There is no instrumented numerical producer bridge implementing BF16
   canonical tile ingestion, actual polar encode, independent arithmetic
   decode/re-encode, canonical I16 symbol emission, FP32 reconstruction and
   the mandatory complete CuPy/NVML receipt.
4. There is no independently authored source manifest, independent source
   review receipt, sealed runtime environment, authenticated source plan or
   output destination authorization.
5. No synthetic numerical smoke, actual three-role pilot, control panel, full
   coarse object, DH384 projection, CAGE oracle or physical composite has run.

These are blocking dependencies, not future details that may be inferred as
passing.

## Boundary decision

Producer verdict: **READY FOR INDEPENDENT SOURCE-ONLY AUDIT OF THE DESIGN AND
GATES; BLOCKED FOR NUMERIC PRODUCER, PAYLOAD AND CUDA**.

The independent auditor should inventory the stopped tree, publish its own
content root, challenge packet and publication canonicality, and decide
whether to author a source manifest. Any runtime-lock or producer-bridge
addition changes the tree and requires a new audit. A future DH384 pilot result
must remain scoped to DH384; it cannot be promoted into a negative claim about
broader TACTIC-CAGE architectures.
