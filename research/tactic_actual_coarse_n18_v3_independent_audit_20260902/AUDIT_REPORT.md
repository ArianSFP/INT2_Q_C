# Independent hostile source audit — TACTIC actual-coarse N18 v3

Date: 2026-09-02  
Producer tree: `research/tactic_actual_coarse_n18_v3`  
Verdict: **BLOCK — do not create a producer source manifest, PASS seal,
runtime freeze, numerical producer, payload run, CUDA launch, or result from
this revision**

This is a source-only verdict. It is not a numerical result for N18-307,
TACTIC-DH384, TACTIC-CAGE, or Qwen. No model file or Qwen weight was opened;
no numeric producer or pinned numeric dependency was imported; no CUDA context
was created; and no reconstruction, Gaussian control, physical codec artifact,
or MSE result was emitted.

V3 repairs several real v2 defects: its source modules execute from the bytes
that were authenticated, the obvious fabricated runtime-lock fields fail, the
publisher verifies staging before its only public rename, dependency IDs and
inventory aggregates are bounded, and its one-pass **unique-page** arithmetic
is correct for declared explicit frames. Those repairs do not yet make an
actual coarse codec. The decisive blocker is that v3 replaces a frozen N18
packet with a byte-allocation identity but supplies no packet language or
decoder for that identity. Its zero-slack metadata and shared-packet topology
are also unresolved, its tiny fallback is outside the target rate/distortion
cell, and its runtime authentication still does not bind the bytes later
imported.

## Exact audited authority

The audit-owned inventory is
[`producer_inventory.json`](producer_inventory.json), SHA-256:

```text
abedc85c236b468f2550361c6c6e539932529595b78ad5978b2880099a891f40
```

Using the producer's declared root preimage — the literal domain
`TACTIC-N18-V3-AUTHENTICATED-SOURCE-ROOT-v1\0`, followed by bytewise UTF-8
filename order and length/hash records — the independent root is:

```text
1db2a1fd9da07743e556a02ce58d97424672129683a09c5b805714b7ce6709f5
```

The exact flat closure is 14 files and 120,570 bytes:

| Member | Bytes | SHA-256 |
|---|---:|---|
| `POSTIMPLEMENTATION_REVIEW.md` | 3,668 | `ff09c53900d0ecdd067e9d8ca8f67abc439e0c71bea66e7571162773aba3e236` |
| `README.md` | 6,144 | `8246bf42efb4dca0a251534e464090e608482f7e2f2c7ed7905a411175688300` |
| `dependency_auth.py` | 7,230 | `7a3a00ce5fd8ba466f0caee10d7a915f589ca8646c6ea08cf2141be618f78657` |
| `dependency_graph.json` | 1,321 | `8a5eacd9e9b5633db1ffd095f684f5ee660b0542a24431c047dc35b5aef11e14` |
| `design_lock.json` | 5,116 | `9d08d16f7dd211b7c9a5192f4201a0ff3ac7fd0630d3f12a923d236e092da206` |
| `dispatcher_contract.py` | 4,015 | `9473479d036347ea1aad08ed4115f39f9de5b86d08de880d4248d7b89268cc74` |
| `immutable_bootstrap.py` | 16,053 | `8296e2a683013a09044511732f8560c1eff064e57378f37091c4047d8fc60917` |
| `runtime_auth.py` | 9,768 | `718e1413a243d6c4913a0d84dc7860177c050f30d977ba67eb3a503079c8f0a7` |
| `safe_telemetry.py` | 6,544 | `767ff6f2fa2637d54cf29954b4cb785ae5983b9b2fdb097aad2cb597bc35ba31` |
| `secure_io.py` | 16,495 | `9aec6513d44a939f39b8703a0ab24aa196cef0656c5b206bec546054ac0ee180` |
| `test_source_only.py` | 25,341 | `9d5bbba86aaa0ca7268b77488ea603dfb4106abf4e91174a15f957d337c96102` |
| `universal_layout.py` | 7,079 | `1cbc2ba594e59d7afe703d440b6014b99d222176b2addd688813a34e45514c67` |
| `v3_common.py` | 4,762 | `e847e04797a14d9ebabebecf9bad930b7fd617e7ed139b5cbd92e21cc71f30ee` |
| `verify_source.py` | 7,034 | `6802b59045684f2876f2b023bdbb77d8cab1bd59bc69b80a219cfd2880d46377` |

The parent dispatcher copied that stopped tree to a fresh RunPod path
`/tmp/tactic_n18_v3_source_audit_20260902`; this auditor did not infer trust
from the transfer. It authenticated every staged byte against the independent
inventory/root before accepting any producer output. The post-replay path was
still the same 14 regular files, with no directory, symlink, `__pycache__`, or
extra member. Transfer metadata such as inode and timestamps is not part of
the source identity and is not promoted as provenance.

## Authenticated POSIX replay

The RunPod replay used Linux `6.8.0-107-generic`, `/usr/bin/python3.12`
3.12.3 with `-I -B`, and a held bootstrap descriptor. Python executed
`/proc/self/fd/9`; the inherited FD, bootstrap SHA-256, external inventory
SHA-256, and expected source root were all supplied separately.

The authenticated verifier passed and emitted the exact receipt in
[`runpod_authenticated_verify.json`](runpod_authenticated_verify.json). It
reported 14 files / 120,570 bytes, the independently pinned root, two pinned
prototype sources, the Qwen layout identity, the odd-shape upper-rate/page
identity, and the explicit no-payload/no-runtime/no-CUDA boundary.

The authenticated hostile suite ran all 29 cases on POSIX with no skips:

```text
28 passed
0 assertion failures
1 error
nonzero suite exit
```

The five publication state-machine checks all passed: constructor faults,
every injected prepublication phase, corrupt staging, both post-rename phases,
and ordinary success. The one error was the symlink-leaf case. The leaf was
correctly rejected by `O_NOFOLLOW`, but `HeldRegularFile` leaked raw
`OSError(ELOOP)` where its own test and public contract expected
`v3_common.ContractError`. The exact summary is
[`runpod_authenticated_tests.json`](runpod_authenticated_tests.json). A
source candidate cannot be sealed while its mandatory exact suite exits
nonzero, even though this particular mismatch is not a symlink bypass.

The independent arithmetic/static harness is
[`hostile_audit.py`](hostile_audit.py), SHA-256
`90f1701edcb56879002dd129bd33a1759253ab81f71a1409b4aa0db8c5420c48`.
Its recorded evidence is
[`independent_hostile_receipt.json`](independent_hostile_receipt.json).

## V2 repair matrix

| V2 blocker / requested check | V3 source finding |
|---|---|
| Authenticated bytes versus imported bytes | **Repaired for producer siblings.** The procfd bootstrap is externally pinned; every member becomes an immutable byte packet; a dedicated loader wins over live paths; preloaded siblings are rejected; bytecode is disabled; and the loaded-module receipts are checked. |
| Inventory aggregate, names and traversal | **Repaired.** Exact 14-name closure, bytewise ordering, per-member, row and aggregate caps, flat names, duplicate-key rejection and no-follow opens are enforced. |
| Dependency IDs and paths | **Repaired at the source parser.** IDs and relative paths are bounded and traversal-safe; dependency source bytes remain held. Dynamic-import behaviour of those large pinned programs is still a future manual/runtime concern. |
| Fabricated runtime lock | **Schema repair passes.** Negative/zero bytes, empty versions, zero file counts and all-zero digests are rejected. Actual runtime execution remains blocked for separate reasons below. |
| Completion/rename state machine | **Source primitive passes POSIX fault replay.** The entire staging tree is rehashed before the sole no-replace rename. Pre-rename faults expose no final tree; post-rename faults can leave only the previously verified tree. |
| Constructor cleanup | **Passes all authored constructor phases on POSIX.** |
| Odd/tiny/unequal upper-rate and unique-page arithmetic | **Arithmetic identities pass, but target/universal codec closure does not.** The zero fallback and missing packet semantics are blocking below. |
| External review authority | **Procedurally improved, mechanically conditional.** No producer issuer/PASS helper exists. The consumer binds action/source/runtime/audit fields, but a held regular FD is not cryptographic proof that another authority created it. A trusted external launcher remains part of the trust base. |
| Telemetry strictness | **Schema is materially tightened, but remains validator-only.** UUID/PCI equality, transfer sums, versions, timings and sampled peaks are checked. The values are not independently measured or signed, and no CUDA run occurred. |

## Blocking findings

### 1. The `1,228 bytes / 4,096 weights` rule is not an N18 codec

The frozen v2 coarse packet was executable and self-describing:

```text
128-byte TACN18C2 header
78,464-byte arithmetic reservoir
--------------------------------
78,592 bytes per 262,144 values
```

It bound role, matrix shape, tile ordinal, valid tail length, deterministic SC
and RHT seeds, FP32 reconstruction scale, logical bit length, payload SHA-256,
algorithm ID, CRC, hard EOF, terminal padding, and canonical decode/re-encode.

V3 observes that `64 * 1,228 = 78,592` and calls each 1,228-byte interval a
coarse microblock slot. Across the exact v3 closure, however, there is no:

- `TACN18C2` magic or versioned replacement;
- 128-byte header language;
- packet packer or parser;
- hard logical EOF or arithmetic capacity rule;
- scale, role, shape, tile, seed, valid-value, digest or CRC field;
- canonical decode/re-encode;
- integer-symbol or reconstruction record binding; or
- numerical encode/decode function at all.

For Qwen, each role happens to contain six complete N18 groups, so the byte
counts can be partitioned back into 18 regions. V3 still never states or
checks that concatenating 64 slots recovers the old packet language. For the
legal `769 x 2051` geometry, each role contains six N18 groups, one residual
complete 4,096-value microblock, and a 259-value tail. V3 assigns bytes to the
three residual microblocks and aggregates the three sub-micro tails, but no
decoder language exists for either object. The arithmetic identity therefore
silently changes an N18 codec into an unspecified new family.

This is not a documentation nit: no independently decoded `307/128` coarse
artifact can be constructed from these sources. The physical rate is a count
of hypothetical bytes, not a finite code.

### 2. The zero-slack metadata ledger has no valid topology

For the six-expert Qwen panel, the frozen DH384 ledger had:

```text
coarse reservoirs       8,487,936 bytes
384-bit fine fields       331,776 bytes
six 512-byte headers        3,072 bytes
one global packet          24,576 bytes
----------------------------------------
container                8,847,360 bytes = exactly 2.5 bpw
```

V3 charges four metadata bytes per full microblock. That yields 4,608 bytes
per Qwen expert and 27,648 bytes over six experts — numerically equal to the
old `24,576 + 6*512` total. Equality of totals does not define placement:

- `layout_panel` puts all 4,608 bytes inside each owner's contiguous frame;
- it declares no global interval or shared pages;
- it does not say where the 16,384-byte selector, 2,048-byte QC table,
  2,048-byte seed fixture, 4,096-byte schema, or each 512-byte expert header
  resides; and
- the selected expert cannot reconstruct a global packet that was merely
  sharded across six other owner frames.

If the packet remains global, its six pages must be read and charged for every
routed expert, restoring the frozen `73/72 = 1.013888...x` topology rather
than v3's claimed `1x`. If it is duplicated in every expert, the zero-slack
2.5-bpw budget overflows. If it becomes a compiled universal constant, that
is a different frozen codec and contradicts the existing requirement that the
selector/QC/fixture bytes be physical. For an arbitrary expert count, the
coincidental six-expert equality disappears entirely.

V3 must serialize one concrete container and show which exact byte interval
owns every global, expert, block and tail field. The current 1x result is only
the page count of an undefined layout.

### 3. The zero-byte fallback is outside the target cell

The fallback is honestly labelled as having no quality claim, which prevents
it from being misreported as a positive result. It does not close the stated
target for arbitrary positive shapes.

For a legal `1 x 1` expert:

```text
weights                  3
physical bytes           0
physical rate            0 bpw
relative MSE of zero     1 for every nonzero source
F at actual R            1
```

Thus it violates both the artifact floor `R >= 2.15` and `F <= 0.8`. A legal
`1 x 4096` expert is more revealing: the partition computes three complete
microblocks with 3,684 nominal coarse bytes, 144 fine bytes and 12 metadata
bytes, then discards all 3,840 bytes because the owner is below the 8,191-byte
threshold and decodes the entire expert as zero.

The fallback can remain an explicit out-of-target compatibility decode, but
then the package must not claim that every accepted positive shape can emit an
eligible target artifact. Alternatively, the eligible shape domain must be
narrowed and enforced before encoding. Merely saying that quality is
unmeasured does not satisfy the rate/MSE gate.

### 4. Runtime hashes are not the bytes later imported

The runtime-lock validator no longer accepts the v2 all-zero/negative
placeholder. It also holds and hashes the interpreter. Distribution handling
still has a hash-to-import TOCTOU:

1. `_tree_distribution` obtains the `importlib.metadata` RECORD file list.
2. It opens, hashes and closes each listed file one at a time.
3. `RuntimeAuthority` retains only the lock and interpreter descriptors.
4. A later dependency import resolves CuPy, NumPy, SciPy and pynvml through
   ordinary Python import machinery.

There is no immutable distribution snapshot/byte loader, no held file set,
no exact installation-directory closure, and no rejection of an already
preloaded numeric module. A package file can change after its hash and before
or during normal import. `reverify_distribution_trees()` is optional and
post-run rehashing would not prevent execute-then-restore attacks.

The supplied RunPod exposes a second operational contradiction:

```text
sys.executable: /workspace/int2-cupy-venv/bin/python
is symlink:     true
real path:      /usr/bin/python3.12
```

`authenticate_runtime` requires the locked path to equal `sys.executable`,
while `HeldRegularFile` uses `O_NOFOLLOW` on the leaf. The intended venv path
therefore raises `ELOOP`; locking `/usr/bin/python3.12` instead fails the
literal path equality when launched from the venv. A future runtime lock is
not executable on the stated environment without changing this contract.

Required repair: bind the running image by `/proc/self/exe`/held inode rather
than a symlink spelling, reject preloaded numeric modules, and execute numeric
code from a held immutable installation snapshot or equivalent loader. A
complete directory/RECORD policy must also reject unlisted importable files.

### 5. Unique pages do not bound repeated reads

`OwnerFrame` reports the union of distinct 4 KiB pages intersecting a
contiguous frame. That is useful and correct. The design contains no decoder
schedule, compressed-frame pass count, kernel traffic receipt, or repeated
read ledger.

At the smallest explicit frame, 8,191 stored bytes can touch 12,288 unique
page bytes at the hostile starting offset:

```text
one-pass unique-page amplification = 1.5001831278232207x
two full compressed-frame passes   = 3.0003662556464414x
```

Even Qwen's page-aligned 1x frame becomes 2x if the physical compressed frame
is fetched twice. A future implementation can keep bytes resident or fuse
coarse/fine decoding, but that must be demonstrated by a concrete decoder and
instrumented traffic ledger. Contiguity alone does not prove one read.

### 6. The mandatory POSIX suite is red

The raw `ELOOP` discussed above is a narrow exception-normalization bug, not a
successful symlink traversal. Nevertheless, the exact mandatory suite exits
nonzero (`28/29`). Source sealing must be fail-closed. Wrap no-follow open
failures in the declared contract exception and replay the exact suite.

### 7. The dispatcher ABI relies on convention, not mechanical provenance

V3 appropriately removes the producer-generated review token and contains no
`issue_*`/PASS constructor. `consume_dispatcher_assertion` binds a held JSON
document to caller-supplied action, source root and runtime-lock hash. It does
not receive an externally pinned assertion digest or signature, and an FD
number `>=3` does not distinguish an inherited dispatcher capability from a
file created by the current process.

This can be a valid ABI when a separately reviewed launcher is explicitly in
the trust base. It is not, by itself, evidence of independent review. The
future numerical entry must be callable only through that launcher and the
audit evidence/launcher identity must be pinned outside the producer.

## What is sound and should be retained

- The externally pinned procfd bootstrap closes the v2 source hash/import gap.
- The exact flat inventory, aggregate cap, path rules and immutable sibling
  loader are strong source-closure primitives.
- The runtime schema rejects all tested fabricated placeholder fields.
- Dependency IDs, paths, AST import-root declarations and held source packets
  are materially improved.
- The verify-before-rename publisher passed every substantive POSIX fault
  phase in the authored suite.
- For defined explicit byte intervals, the integer upper-rate decomposition
  and one-pass unique-page union are correct: Qwen is exactly 2.5 bpw / 1x by
  that limited metric, and `769 x 2051` is 2.4999986262740514 bpw /
  1.000009468147124x.
- The documentation correctly says there is no Qwen payload result, runtime
  freeze, CUDA run, or MSE claim.

## Required next revision

Build a new sibling; do not patch or seal this root. It should:

1. choose and implement one versioned finite coarse packet language;
2. either preserve the exact 78,592-byte N18 header/reservoir semantics for
   every complete group or explicitly freeze a new microblock codec and prove
   its arithmetic capacity;
3. define residual-N18-group and aggregate-tail encode/decode/re-encode rules;
4. map the 24,576 global bytes, 512-byte expert headers, 48-byte fine fields
   and all tail/framing/check bytes into literal offsets under one ledger;
5. distinguish out-of-target fallback shapes from shapes eligible for an
   `R in [2.15,2.5], F <= 0.8` claim;
6. bind the actual runtime import bytes through execution and resolve the
   venv-symlink contradiction;
7. emit unique-page **and repeated-pass/traffic** ledgers from a concrete
   decoder schedule;
8. normalize no-follow errors so the mandatory POSIX suite is green; and
9. keep review authorization in a separately authenticated dispatcher.

Only after those source repairs should an external audit create a new
inventory/root and consider runtime freezing. The actual N18 coarse artifact,
frozen DH384 pilot, broader CAGE tests, Qwen payload, CuPy execution and MSE
claim all remain downstream and unauthorized by this report.
