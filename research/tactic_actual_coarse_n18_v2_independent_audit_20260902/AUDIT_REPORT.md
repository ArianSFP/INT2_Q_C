# Independent hostile source audit — UNIPOLAR-N18-307 v2

Date: 2026-09-02  
Producer tree: `research/tactic_actual_coarse_n18_v2`  
Verdict: **BLOCK — do not create a source manifest, review receipt, runtime
freeze, numeric producer, payload, or CUDA launch from this revision**

This verdict is source-only. It is not a negative result for TACTIC-DH384 or
TACTIC-CAGE. No model weight, Qwen payload, numerical reconstruction, Gaussian
control, CUDA context, lower-rate artifact, result envelope, or physical
composite was opened or created.

## Exact pre-review authority

Before importing a producer sibling, the independent audit checked all 12
top-level regular files by exact byte count and SHA-256. It independently
recomputed the producer-declared pre-review root as:

```text
5719a483eef05571e93ab53eca80563a1a90a2c30d72644498fe0355735be917
```

The preimage is the literal UTF-8 prefix
`TACTIC-N18-V2-PRE-REVIEW-ROOT-v1\n`, followed by records in
case-insensitive filename order:

```text
name<TAB>decimal_bytes<TAB>lowercase_sha256<LF>
```

The exact inventory was:

| Member | Bytes | SHA-256 |
|---|---:|---|
| `dependency_graph.json` | 1,745 | `0751a0a64ab18e9195508bfdfcf05edd11d43e4120e37036dfeb6ec4ded2d355` |
| `design_lock.json` | 4,482 | `be8fffe0eee387d07227ef6361fe78abc8e1ef0f901c8ba40b3184804f1a133c` |
| `n18_common.py` | 16,189 | `61c2719d7c58a968bee8bdcf456d9b6aa9402d134032ec7204c2015580f4dc0c` |
| `POSTIMPLEMENTATION_REVIEW.md` | 6,710 | `17ed841e6db0da8950b5688340eb460053b8e96a73b8af7768656636645a71b6` |
| `preflight_gate.py` | 11,291 | `36673ef80e0035e440dc36e275320b9e96f3636bc54f0c307354b2a5ff942dcf` |
| `README.md` | 8,011 | `2920fdf0dd29e475a0e95776d954071bcc3ae1a3a2d22cbdcd8a989452dd01c9` |
| `runtime_contract.py` | 12,156 | `1ef3c10a37de8366bcf72281dbf2e4aa40184e78ab74cc61dcf322089a755f1c` |
| `runtime_environment_lock.json` | 512 | `79e72bba553ff09eb5a7e1a29d1082a1beafb25eba96fd7092ad701454855be0` |
| `secure_io.py` | 14,211 | `8dc7da600badce77769db529d9e53df5e5667c16ddf4828fc074a6c505beb8cd` |
| `source_adapter.py` | 8,356 | `48c672419c2b20dd36651055c01c3d94c2c3c5417c4f5edf5736c3d0812691eb` |
| `test_source_only.py` | 16,779 | `967dc7c8c3615a16e3fb1de69b4e86005e8ccfa261e1bc702837b527a8b6b73a` |
| `verify_source.py` | 11,732 | `e1b808fe5c75391ff7fcacaa390caa62a8a21d93aa7062be154a5a0f63e02680` |

The RunPod review path also contained a generated `__pycache__/` directory
with seven `.pyc` files. It is not in the declared inventory and was treated
as untrusted residue. The independent replay redirected Python cache lookup to
an absent audit-only prefix and disabled bytecode writes. No cached producer
bytecode was imported. That review directory is not manifest-ready while the
extra directory remains; the audit did not delete or modify it.

## Executed checks

### RunPod

On the supplied RunPod, after root authentication and cache exclusion, the
producer's selected standard-library suite was replayed with
`/usr/bin/python3.12 -I -B`:

- exact full/tail packet and ledger checks;
- hard EOF, canonical bits, fill, reserved bytes, digest and repaired-CRC
  terminal-pad attacks;
- shape/role portability, Down transpose, identity-field rejection and hostile
  dimensions;
- source-first state transitions and the DH384-only non-converse;
- deliberately unfrozen runtime rejection and review/action binding;
- POSIX held-FD leaf/ancestor symlink rejection;
- basic completion-last/no-replace publication; and
- the exact two-source AST/import graph.

The first invocation used the wrong audit-side repository root for the two
prototype dependencies, producing one `FileNotFoundError`. Repointing it to
the actual `/workspace/INT2__compression` root made that exact dependency test
pass. The resulting producer-method outcome was 20/20 passed; timings were
0.327 seconds for the panel and 0.052 seconds for the corrected dependency
method.

A separate preflight invocation, with the current absent manifest and
unfrozen boundary, returned nonzero while the requested output, review path
and source-plan path all remained absent. It stopped at missing
`SOURCE_MANIFEST.json`; no payload or CUDA path was reached.

### Independent harness

`hostile_audit.py` authenticated the exact root before producer imports. Its
Windows standard-library replay ran 16 tests in 2.363 seconds: 11 executed and
passed, while five explicitly POSIX-only fault-injection cases were skipped.
The executed cases include constructive counterexamples for the runtime-lock
validator, telemetry validator, legal-shape rate/read cap, source bootstrap
data flow, manifest materialization bound and dependency-ID parser.

The independent POSIX fault-injection harness was not copied to the external
RunPod because the environment denied export of local audit source. Therefore
the post-rename and constructor-fault findings below are exact constructive
source traces, not claims of a completed independent dynamic replay. The
producer's ordinary POSIX success-path tests did run and pass.

## What is sound in this snapshot

The following source-level mechanisms survived review:

- A full N18 reservoir is exactly 78,592 bytes. For a full 262,144-value tile,
  its physical rate is exactly `307/128 = 2.3984375` bpw.
- The 128-byte header field layout is internally consistent and binds role,
  shape, tile, valid tail, shape/role/coordinate-derived seeds, transmitted
  FP32 scale, logical length, payload digest, algorithm identifier and CRC.
- The logical bit language is MSB-first, rejects zero extension, enforces hard
  EOF and zero terminal/fill bits, and consumes a hostile iterable with a
  finite 627,712-bit cap.
- Full fixed reservoirs are charged for tails; the implementation does not
  hide tail bytes.
- Gate/Up/Down shape rules and Down-transposed canonical coordinates are
  shape-portable. Seeds exclude expert/model/checkpoint/layer identity.
- Counts and products in the parsed source plan are bounded before source
  files open. Pilot mode selects only one coordinate-first triplet.
- The owner-aware **byte** ledger correctly avoids the v1 equal-share error.
- Held input files walk every absolute component with `O_NOFOLLOW`, retain a
  leaf FD, stream hashes in 1 MiB chunks and recheck content/identity.
- The exact current runtime placeholder really does fail before source plan,
  model payload, numeric import or CUDA.
- The source-first protocol stops frozen DH384 only. It does not falsely turn
  a DH384 failure into a converse for CAGE.
- The documentation is honest that no numerical producer, actual coarse
  stream, Qwen result or CUDA authority exists.

These are useful design-scaffold properties. They do not overcome the
blocking closure failures below.

## Blocking findings

### 1. Authenticated source bytes are not the bytes imported

`preflight_gate._bootstrap_source` opens and hashes each package member, stores
its bytes in `_packets`, closes every member FD, closes the package-directory
FD, and returns. `main` then inserts the live package path into `sys.path` and
imports `n18_common`, `runtime_contract`, `secure_io` and `source_adapter` from
that path.

There is no immutable source snapshot and the authenticated `_packets` are not
compiled or executed. A replacement between `_bootstrap_source` returning and
the first sibling import executes bytes that were never hashed. This is a
direct hash-to-import TOCTOU and contradicts the claimed held-source
auth-before-import boundary.

Required repair: execute the already authenticated bytes in an isolated module
namespace, or write them to a private no-follow snapshot and import only from
that snapshot while holding its directory. Disable or redirect bytecode cache
lookup. Do not attempt to repair this by rehashing after import, because module
top-level code has already executed by then.

### 2. The runtime "authentication" validator does not authenticate runtime

`validate_environment_lock` checks JSON shape, a self-hash, the literal
`sys.executable` path, digest syntax and distribution names. It never stats or
hashes the interpreter and never enumerates, RECORD-verifies or hashes any
installed distribution file.

The independent counterexample supplied:

- interpreter `bytes = -1`;
- interpreter SHA-256 of 64 zeroes;
- empty Python and distribution versions; and
- zero RECORD/tree hashes for all four required distributions.

After recomputing only the JSON self-seal, the validator accepted this lock.
Thus a future status change to `FROZEN_AUTHENTICATED_RUNTIME_READY` would not
establish the runtime claimed by its name.

Required repair: use held no-follow descriptors to check interpreter
type/size/hash and deterministically verify every installed distribution's
RECORD plus complete file-root digest before returning. The verified values
must be compared to the lock, not merely checked for SHA-256 syntax.

### 3. A post-rename verification failure leaves a public COMPLETE tree

`CompletionLastPublisher.complete` writes `COMPLETE.json`, fsyncs staging,
renames staging to the public final name, fsyncs the parent, and only then
reopens and rehashes the public files. `self.finished` is set only after that
rehash.

Constructive failure trace:

1. Public `result/` is absent.
2. `complete()` writes the index and `COMPLETE.json` in staging.
3. `renameat2(RENAME_NOREPLACE)` makes `result/` public.
4. The public `payload.bin` is corrupted without changing its size.
5. The real `_hash_all` detects a SHA-256 mismatch and `complete()` raises.
6. `result/COMPLETE.json` and the corrupt public payload remain visible.
7. `abort()` targets the old staging name, which no longer exists, and can
   raise `FileNotFoundError` instead of removing or quarantining the public
   tree.

This is complete-looking-but-operation-failed, not complete-or-absent.

Required repair: fully enumerate and rehash the held staging directory before
the single publication rename. If a post-publication check is retained, use an
explicit phase machine and a safe quarantine protocol so an exception cannot
leave a trusted completion marker beside unverified bytes.

### 4. Legal universal shapes can exceed both target caps

The format is syntactically shape-portable, but its fixed full reservoir per
partial tile is not target-rate portable. For the package's own legal
`769 x 2051` portability geometry:

```text
coarse-only physical rate = 2.790450787113267 bpw
```

That already exceeds the final 2.5-bpw cap before any 384-bit refinement or
metadata. In an unequal panel containing legal `1 x 1` and `768 x 2048`
experts, the owner-aware byte ledger gives the tiny expert a cold
amplification of exactly `224695x`, far above the `<2x` contract.

The fixed N18 Qwen geometry happens to be exactly tiled; that does not close a
universal SwiGLU-MoE codec. A shape-derived tail/fallback packet and expert
layout must guarantee the physical cap for the artifact being promoted, or
the supported geometry domain must be narrowed honestly. Qwen-only exact
divisibility cannot be hidden in a universal claim.

### 5. The review receipt is self-asserted, not independently authenticated

The action strings are public constants. `validate_review_receipt` accepts a
caller-provided JSON document whose only seal is SHA-256 of that same document
after removing the seal field. No externally pinned receipt digest, signature
or reviewer key is checked. The preflight caller also chooses the receipt path.

This can be a procedural acknowledgement, but it does not mechanically prove
an independent review. If the gate is intended as an authority boundary,
require an externally pinned receipt digest or signature tied to the manifest,
action and findings. This audit deliberately issues no PASS receipt.

## Additional repair findings

### Source-manifest bootstrap can materialize an unbounded aggregate

The manifest file is capped at 1 MiB and each member at 1 MiB, but
`_bootstrap_source` does not enforce the exact 12-name set, a row-count cap or
an aggregate member-byte cap before filling `packets`. `verify_source.py`
contains a stricter expected-set check, but preflight does not call it and is
the actual auth-before-import path. Enforce schema, exact names, unique sorted
rows and total bytes before opening the first member.

### Dependency snapshot IDs permit path traversal

`authenticate_dependencies` validates `relative_path` containment but does not
validate `row["id"]` before constructing
`os.path.join(snapshot.name, f"{row['id']}.py")`. The exact checked-in IDs are
safe, so this is not an exploit of the authenticated snapshot; it is a parser
defect that must be fixed before accepting a future dependency graph.

### Constructor failure leaves hidden staging residue

If `os.mkdir(staging)` succeeds and the following staging-directory `os.open`
fails, the constructor closes only `parent_fd`. The private hidden staging
directory remains. It is not a public final artifact, but repeated faults can
leak residue and it violates clean abort semantics.

### Read accounting is byte-level, not page-level

The equal Qwen ledger's `1x` is exact for owned reservoir bytes. It is not an
exact cold-page union. For six contiguous 1,414,656-byte expert regions in one
file, 4 KiB page intersections yield a worst owner read of:

```text
1,421,312 / 1,414,656 = 1.0047050307636627x
```

This remains comfortably below 2x, but documentation should call `1x` a
native-byte result and a future artifact must emit an instrumented page union.

### Telemetry schema accepts semantically empty receipts

The exact-field checker accepts empty equal CUDA/NVML UUID and PCI strings, a
string-valued `nvml_physical_index`, empty runtime versions, zero wall/kernel
times and `model_h2d_bytes > logical_h2d_bytes`. Exact keys are not sufficient
telemetry authentication. Tighten types, nonempty formats, cross-field
inequalities and phase/transfer provenance before a numeric bridge is reviewed.

## Boundary decision

This revision is an **honest draft source-only design scaffold**, not a
misrepresented numerical result. Its current missing manifest and deliberately
unfrozen lock make payload and CUDA entry genuinely impossible under ordinary
execution. However, it is **not a valid sealed or authorizable source closure**:
the bytes imported are not the bytes authenticated, the future environment
lock can be fabricated, publication can leave a corrupt public COMPLETE tree,
and the claimed arbitrary-shape packet does not preserve the target rate/read
caps.

Do not author `SOURCE_MANIFEST.json` or any PASS receipt for this root. Repair
the source-import snapshot, runtime verification, publication state machine and
universal tail/layout path in a new producer revision; then recompute a new
pre-review root and repeat an independent POSIX fault audit. The numeric
producer, actual N18 coarse artifact and DH384 source pilot remain downstream
work and are not authorized by this report.
