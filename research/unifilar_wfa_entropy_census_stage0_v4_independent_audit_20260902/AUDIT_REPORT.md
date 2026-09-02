# Independent hostile source audit — UWFA-SC v4

Date: 2026-09-02  
Producer: `research/unifilar_wfa_entropy_census_stage0_v4`  
Verdict: **BLOCK — do not create the producer manifest/runtime freeze or open a Qwen/current-codec/control payload from this revision**

This is a source-only boundary verdict, not a negative scientific result for
long-range WFA/MPS modelling, TACTIC-CAGE, or Qwen. The audit did not open,
stat, hash, enumerate, or infer any model/Qwen weight, current codec object,
extracted decision stream, or matched-Gaussian payload. It did not edit the
producer, create a producer manifest/result, run a numeric producer, commit, or
make a production claim.

## Exact authority

Before any producer module was compiled, `hostile_audit.py` rejected symlinks
and nonregular members, read every member through a no-follow held descriptor,
checked stable fstat identity, exact size, and SHA-256, and captured the bytes
used for snapshot compilation. The exact 17-member root is:

```text
57d7c99e616da8d56dabb7fedab75fb8a9dbf940008762d1e43ae452ad4356c6
```

It is SHA-256 over the literal prefix
`UWFA-SC-V4-INDEPENDENT-PRE-REVIEW-ROOT-v1\n` followed by
case-insensitive-name-sorted UTF-8 records
`name<TAB>decimal_bytes<TAB>lowercase_sha256<LF>`. The complete inventory is
in `producer_inventory.json`. `SOURCE_MANIFEST.json` was correctly absent.

The RunPod copy `/tmp/uwfa_v4_dev_20260902b` was enumerated and hashed before
imports. All 17 names, sizes, and digests equal the local tree byte-for-byte;
there were no extra members. Python ran isolated with `-I -B` and bytecode
writes disabled. The producer's own source-only suite ran 48 tests in 73.436
seconds: 47 passed and the sole manifest-absent pre-review test skipped as
expected.

## Decisive blockers

### B1 — completion can publish bytes it did not commit

`CompletionLastOutput.complete()` checks only that its caller returns the same
in-memory metadata list produced by `write_new()`. Immediately before writing
`COMPLETE.json` and renaming, it does not re-enumerate the held staging
directory and does not re-open/re-hash its declared members.

The POSIX counterexample wrote `RESULT.json`, mutated it through the held
staging directory, added undeclared `UNDECLARED.bin`, then called `complete()`.
The producer published the final directory and `COMPLETE.json`. Only a later
call to the independent envelope verifier detected the inconsistency. Hidden
mode `0700` does not protect against same-UID producer faults or accidental
staging writes. Publication must verify exact staging membership and held-file
bytes immediately before completion, or retain sealed member descriptors that
cannot be altered.

The v3 symlink-ancestor defect itself is repaired: output ancestry is opened
no-follow, the parent identity is retained, writes are descriptor-relative,
and publication uses `renameat2(RENAME_NOREPLACE)`. A parent-path substitution
test published under the retained original inode, not the replacement path.
The new B1 is narrower but still blocks a completion-last authority claim.

### B2 — controls are not bound to the source closure

`controls_phase()` reads source full/structural geometry, pipeline, and panel,
but ignores `source_result["_bindings"]`. For each control it checks only the
control evidence's pipeline and its own full/structural geometry. It does not
require equality with the source for:

- universal decoder;
- producer manifest or external bootstrap;
- extraction program or universal adapter;
- authenticated source snapshot root;
- source-free preflight receipt; or
- baseline plan.

The separately supplied `source_artifact_sha256` is also trusted without being
compared with the actual source panel's authenticated artifact digest.

The hostile test used real sealed control-binding records and real sealed score
records. It supplied an artifact digest different from the source panel and a
foreign decoder/manifest/bootstrap/extraction/adapter/snapshot/preflight
closure for all eight controls. Every existing interface check accepted them;
only the later numerical fit/pack functions were mocked to keep the test
source-only. This violates symmetric source/control provenance and permits a
different decoder/runtime closure to define the null statistic.

### B3 — the typed source-free preflight validator does not prove its claims

`validate_source_preflight()` checks a list length of 150 and that every row
has `repeated_gpu_run_exact=True`, but it does not require the canonical
selector set `0..149`, unique cells, candidate metadata, exact result hashes,
or logical-length receipts. It also accepts a very sparse representative record
without binding the winning candidate, container, fixture source, candidate
coverage details, or deterministic result fields.

The counterexample supplied 150 copies of selector 0 with the pass boolean,
plus the minimal accepted representative booleans. After honestly recomputing
the outer receipt seal and `BoundEvidence` digest, the producer validator
returned success. A genuine external runner currently emits much stronger
evidence, but the production input boundary does not enforce that evidence.

## v3 blocker regression disposition

| v3 issue | v4 source finding |
|---|---|
| Fold grammar failed on one-layer/many-expert panels | **Repaired.** Exact-identity exclusion leaves 15 development streams for each six-expert fold; coordinate-disjoint splitting is typed, nonpromoting, and unestimable. A legal one-expert panel stops before fitting. |
| Source geometry/preflight not gated | **Source geometry repaired; preflight only partial.** Full and structural source geometry bind before fit, UUID/PCI checks exist, but B3 shows receipt semantics are under-validated. |
| Output followed a symlink ancestor | **Repaired**, but superseded by B1 staging-integrity failure. |
| CUDA maxima exceeded budgets/no dynamic gate | **Repaired.** Static `MAX_PACKED_SYMBOLS` derives from the 28-GiB cap, and dynamic RSS/free-VRAM admission occurs before blob concatenation/CuPy allocation. |
| CUDA receipt lacked UUID/PCI join | **Repaired.** CUDA and independently sealed `nvidia-smi` UUID, PCI bus, and device name must agree; telemetry failures are fatal. |
| Control geometry equated source-dependent lengths | **Repaired.** Cross-source structural geometry excludes payload/logical lengths, while each source has its own full geometry. B2 is a different closure-binding failure. |
| Posterior handoff mislabeled bit-only digest | **Repaired.** The commitment is explicitly length-prefixed `(bits, levels, base-u16le)` and changes when any triplet component changes. |

The frozen candidate bank is canonical: 150 unique selector ordinals `0..149`,
integer Jeffreys Q0.16 rounding/clamping is replayed, and every serialized model
round-trips byte-for-byte. The source-first gate still prevents controls from
opening after an absolute source failure. There is no continuous-span
dominance claim in this WFA producer; that TACTIC-DH384 question belongs to its
separate blocked audit and receives no credit here.

## Literal rate, portability, and routed-read checks

- Physical rate is computed from literal `8*len(container)` divided by the
  semantic shape-derived source weight count. Header, semantic packet, model,
  directory, frame, alignment, padding, and owner regions are all in the byte
  ledger. Synthetic bits/symbol are not used as physical rate.
- Exact owner-local total and nonpadding denominators conserve every literal
  byte. The cold gate uses the unique 4-KiB touched-page union and the worse
  denominator. Installation authentication is separately reported.
- Every routed byte range is retained, allowing an independent auditor to sum
  repeated requested bytes. The producer does not emit a separate repeated-I/O
  aggregate; this is a reporting limitation, while the frozen cold metric is
  explicitly unique cold pages.
- Symbol density is not an explicit receipt field, but is exactly derivable as
  directory symbols divided by shape-derived weights. The independent two-
  expert fixture produced `0.03466796875` symbols/weight; each route requested
  8,871 literal bytes and touched 20,480 unique page bytes.
- The E250 unequal-shape fixture passed with 250 experts, 751 streams, 251
  regions, a shared Gate tail owned by experts 0 and 249 with unequal one/two
  scalar contributions, exact scalar coverage, and high owner bits intact.
- The producer suite's unselected-private-frame corruption case confirms that
  selected routed decode does not read the corrupt frame/page, while full parse
  rejects it.

These are source mechanics and portability facts, not useful-rate or Qwen
results.

## Two independent source-free RTX 5090 replays

The source boundary explicitly authorizes development-only source-free CuPy
replay. Two fresh isolated executions were run on the supplied RTX 5090. An
audit-owned verifier then strictly parsed both raw receipts and completion
records, recomputed identity, preflight, source-inventory, member, and
completion seals, required all 150 unique cells, checked measured H2D/D2H and
fatal telemetry categories, and compared deterministic results.

Both runs reported:

- device UUID `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722`;
- PCI bus `00000000:16:00.0`;
- development source root
  `d11fa4af39dd5a3c26756cf6b87b201ce70d5745d6c0e0c45878236520dd943d`
  (the runner's canonical-JSON inventory encoding of the same exact 17 files);
- 150/150 unique candidate cells with exact repeated CPU/CuPy equality;
- identical per-cell count tensors, fitted Q0.16 tables, and logical lengths;
- representative winner suffix, 2 states, reset 4096, selector 4;
- identical representative container
  `89c6e81f5cf47e8502cc25c64fb1aa612caa4ec765e405336d1661a5c2951b87`;
- identical full-panel logical lengths and 224,366,256 measured updates; and
- passed runtime/resource/telemetry gates.

Raw receipt SHA-256 values are
`f373a0e969b88f044b6ee62c9bdcd184c6ea18cd5e75d212f7129e699b805886`
and
`015ee68234e3395c9c23bc04c9500a1647d3d38c08c0739962a877ef28b19a95`.
The independent GPU receipt-audit seal is
`feaf995d0f877e2017dcec1cd81329332ad965945f3b8e320f3845067861a8a2`.

These are explicitly direct-copy, source-free development receipts. They grant
no payload authority and cannot override the three source blockers.

## Executed independent evidence and limitations

The authenticated POSIX hostile harness ran 11/11 tests successfully in 5.591
seconds. “Success” means the expected repairs held and each decisive hostile
counterexample was reproduced. Its internal receipt seal is:

```text
da60dabb634ff5709de341c94128bacdb0a2638d93946e0a2ad63e426edf767c
```

No local Python runtime was available on the Windows host, so the complete
independent suite ran on the byte-identical RunPod snapshot. This is sufficient
for the POSIX publication and source-only CUDA boundary being audited; the
exact harness and receipts are included. No Windows reparse-point production
bootstrap was claimed. The future external bootstrap and result auditor remain
separate mandatory audits after these blockers are repaired.

## Required repair before re-review

1. Re-enumerate the held staging directory and re-open/hash every declared
   member immediately before `COMPLETE.json`, rejecting mutations and extra
   names, or use an equivalently sealed descriptor design.
2. Derive the source artifact digest from the authenticated source result and
   require every control's complete decoder/manifest/bootstrap/extraction/
   adapter/snapshot/preflight/plan closure to equal the source closure where
   symmetry requires it.
3. Validate exact canonical all-150 selectors and full per-cell receipts, plus
   a strict representative schema binding candidate coverage, winner,
   container/rebuild, fixture, deterministic hashes/lengths, resource plan,
   and measured telemetry.

Then create a new sibling revision and repeat the exact pre-import inventory,
producer suite, hostile counterexamples, and two source-free GPU replays. Do
not patch this audited producer in place and reuse this receipt.
