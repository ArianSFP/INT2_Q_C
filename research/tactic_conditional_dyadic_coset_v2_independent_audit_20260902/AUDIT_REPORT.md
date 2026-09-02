# Independent hostile source audit — TACTIC-DH384 v2

Date: 2026-09-02  
Producer tree: `research/tactic_conditional_dyadic_coset_v2`  
Verdict: **BLOCK — do not authorize synthetic CuPy, actual-coarse payload,
Qwen access, or a finite-codec claim from this revision**

This is a source-only verdict. It is not a negative distortion result for the
rank-384 conditional dyadic hypothesis, TACTIC-CAGE, graph lifting, posterior
reconstruction, or Qwen. No model, Qwen weight, coarse payload, CUDA context,
numeric compression result, or result container was opened or created.

## Exact source authority

The independent audit authenticated every one of the producer's eight
top-level regular files before executing any producer byte. Producer modules
used by the hostile harness were compiled directly from the authenticated byte
strings; the producer directory was not placed on `sys.path`, cached bytecode
was not read, and bytecode writes were disabled.

The independently defined root is:

```text
cd03644f0e1c36f1c568208d863c10bfd52959fb3dd5e47b6d5c41132dafb61d
```

Its preimage is the literal UTF-8 prefix
`TACTIC-DH384-V2-INDEPENDENT-SOURCE-ROOT-v1\n`, followed by records in
case-insensitive filename order:

```text
name<TAB>decimal_bytes<TAB>lowercase_sha256<LF>
```

| Member | Bytes | SHA-256 |
|---|---:|---|
| `cupy_preflight.py` | 4,764 | `db358f080a8d77d27204f7d098fe1caac65e172edaf17ab6ca205900b5883553` |
| `design_lock.json` | 8,045 | `6549003de0a33797baab9131eb02c720389541272aa44902d65b950ae29f1c9a` |
| `README.md` | 12,502 | `638d3688e638fb1dd82d68b387acc7c97eb25235073c1abc5f85f5af929a8c08` |
| `SOURCE_MANIFEST.json` | 1,291 | `f8de593784638cf7719d08ddda7061f4912166021214fb7a2894862a53050662` |
| `stage0_gate.py` | 21,006 | `3bae2633ac38cef3db12c3327f967ae1d3bc2b7caab0639df92bec5a009288a3` |
| `tactic_v2_common.py` | 14,619 | `1d007e47f075d7b4c746d53e5bffb999ebde4cc1a2b85835cf44e951c18c87ba` |
| `test_source_only.py` | 5,647 | `367e47b62b6c5c3474626b026f6c995de50183a9fc37711fc3de853c48675177` |
| `verify_source.py` | 17,144 | `34c43105fec9acfa4bf834b4c8c7f7ccb668111d50c5f8f76070caaccbfe9087` |

The machine-readable inventory is in `producer_inventory.json`.

## Executed evidence

On the supplied RunPod, a pre-import filesystem inventory found exactly those
eight regular files and no `__pycache__` residue. SHA-256 was recomputed for
every member before the producer's documented standard-library commands ran.

The producer's own verifier returned
`PASS_SOURCE_ONLY_NO_EXECUTION_AUTHORITY`, and its source suite passed 11/11
tests in 0.038 seconds. The same verifier fails under Python isolated mode
because it relies on importing `tactic_v2_common` from the live script
directory; its documented non-isolated `-B` invocation passes. The replay
record is `producer_source_replay_runpod.json`.

The independent `hostile_audit.py` then ran with `/usr/bin/python3.12 -I -B`.
It passed 12/12 checks, with no skips, failures, or errors in 0.043 seconds.
Those checks include the POSIX public-partial-output fault, a constructive lock
with all 108 reservoir rows aliased to one path, an independently generated
selector packet, independent exact rate arithmetic, a transform norm/projection
check, and static control-flow/authentication checks. The receipt is
`hostile_audit_receipt_runpod.json`; its internal receipt digest is:

```text
b27a576efabe31a8caedf42ece81656a1536b15ce623fe14b9ff7ca6dc972fcc
```

`PASS_AUDIT_HARNESS_BLOCK_PRODUCER` means that the audit's expected sound and
unsound properties were reproduced. It is not a producer PASS.

## What is sound

### Exact fixed-geometry arithmetic

For the declared six-expert, `768 x 2048`, three-role object:

```text
weights per expert              = 4,718,592
coarse bytes per expert         = 18 * 78,592 = 1,414,656
fine bits per 4,096 block       = 384 = 48 bytes
fine bits per expert            = 1,152 * 384 = 442,368
fine bytes per expert           = 55,296
expert frame                    = 512 + 1,414,656 + 55,296
                                = 1,470,464 bytes = 359 pages
global packet                   = 24,576 bytes = 6 pages
container                       = 24,576 + 6 * 1,470,464
                                = 8,847,360 bytes
```

The rate decomposition is exactly:

```text
coarse     307/128 = 2.3984375 bpw
fine        12/128 = 0.09375 bpw
metadata     1/128 = 0.0078125 bpw
total       320/128 = 2.5 bpw
```

There is zero uncharged rate slack. If the declared layout is actually emitted
and one selected expert is read once, the page calculation is also exact:

```text
(6 global pages + 359 expert pages) / 360 equal-share pages
= 73/72
= 1.0138888888888888x.
```

### Universal selector and continuous transform

The independent SplitMix64 implementation regenerated the 3,072 active
selector bytes and the canonical 16,384-byte padded packet. Its SHA-256 is the
declared:

```text
0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad
```

The selector is fixed at ordinal 17. No selector candidate loop, fitting API,
model identity, layer identity, expert identity, checkpoint provenance, or
external table was found in the frame construction.

For a fixed decoded symbol block, each stage is a signed permutation followed
by pairwise Hadamards. Twelve stages and the final `1/64` factor form an
orthogonal transform. The independent check reproduced energy preservation and

```text
||e - B B^T e||^2 = ||e||^2 - ||B^T e||^2.
```

Therefore the implemented continuous rank-384 projection is a valid
source-leaking upper bound for any *already defined* finite correction of the
form `B a`. This audit preserves that oracle; it does not elevate it to an
emitted codec.

### Source-first controls and held payload descriptors

The main path opens and evaluates panel 0 (`source`) first. Control payloads
are opened only when source is not a hard reject. The held payload helpers use
component-walk `O_NOFOLLOW`, regular-file `fstat`, same-descriptor hashing and
reads, and retain child descriptors until the held root closes. These are good
primitives, subject to the lock-binding defect below.

## Blocking findings

### 1. The claimed 384-bit finite codebook is not frozen or implemented

The README says that a fixed QC/trellis maps each 384-bit block record to a
bounded Q12 vector and that one public rational output scale is applied. No
such mapping, trellis, coefficient parser, QC table, output scale, finite
encoder, or independent finite decoder exists in any authenticated source
file.

The ledger reserves 2,048 bytes for “Q12/QC tables,” but no canonical 2,048
bytes are constructed, serialized, hashed, or consumed. `stage0_gate.py`
writes only the selector packet and a JSON continuous-oracle report. It never
constructs the claimed 24,576-byte global packet, an expert frame, or any of
the `1,152 * 48` fine records.

Consequences:

- the set of `2^384` legal reconstructions is undefined;
- the output scale can still be chosen after observing results;
- continuous-span dominance cannot be checked against a literal finite map;
- the zero-slack 2.5-bpw container is arithmetic, not an emitted artifact; and
- DH384 cannot yet be the frozen fine backend required by TACTIC-CAGE.

Required repair: specify canonical bytes for the QC/trellis and scale, freeze
their hashes before source access, implement independent encode/decode, prove
every finite correction lies in the audited span, and emit/round-trip literal
48-byte records inside the exact reserved fields.

### 2. The source verifier does not authenticate the bytes it executes

`verify_source.py` imports `tactic_v2_common` at module top level before
`verify_manifest` runs. `stage0_gate.py` also imports
`tactic_v2_common` before it invokes `verify_package`. Both imports come from
the live script directory. The later manifest pass uses path-based
`read_bytes`, `read_text`, `stat`, and `sha256_file`; descriptors are not held
from hash through execution.

There is also no externally supplied or compiled expected source root or
manifest digest. A coherent rewrite of the source files and their manifest can
pass, and a replacement between verification and a later live-path use is not
prevented. Recording the observed manifest hash in the result after execution
does not create prior authority.

Required repair: begin with a minimal externally pinned bootstrap, open every
source member no-follow, hash and retain exact bytes, and compile/execute only
those authenticated byte strings (or an immutable private snapshot). The
independent harness demonstrates this pattern. Disable/redirect all bytecode
lookup and keep the expected root outside the mutable package being checked.

### 3. The coarse lock does not bind reconstruction and symbols to coarse bytes

The stage treats the reconstruction and canonical integer symbols as
decoder-visible inputs, so their exact derivation from the 307/128-bpw coarse
reservoirs is security- and validity-critical. The accepted round-trip receipt
is only required to contain:

- `status == PASS_ACTUAL_COARSE_DECODE_REENCODE`;
- `actual_bpw == 307/128`; and
- `coarse_container_bytes == 6 * 18 * 78,592`.

It does not bind panel identity, reservoir hashes, decoder source root,
runtime, decoded reconstruction hashes, decoded symbol hashes, canonical
re-encode hashes, or the complete record table. The lock's seal is merely a
self-hash.

The independent constructive lock also proved that validation accepts:

- all 108 logical reservoir records pointing to one identical path;
- all 18 source records pointing to one source path;
- all 18 reconstructions pointing to one reconstruction path; and
- all 18 symbol records pointing to one symbol path.

Thus a formally accepted lock can provide arbitrary source-derived symbols or
reconstructions unrelated to its opaque coarse streams. Those free values can
program the conditional frame and residual. Held-FD hashing establishes the
files' identities, not the missing decode relationship.

Required repair: require unique canonical record identities/paths (or
explicitly ledgered aliases), and a separately authenticated producer receipt
that commits to every input stream, exact decoder executable/runtime, every
decoded reconstruction/symbol digest, canonical re-encode digest, panel
identity, source/control construction, and overflow result. The gate must
recompute and compare those bindings, not trust a status string.

### 4. A fault leaves a public partial output tree

`HeldOutput` creates the caller's final output directory immediately. It has no
private staging name, completion marker, file index, final enumeration, final
rehash, or atomic no-replace publication step. In stage 0 the directory is
created before the held coarse root opens and before CuPy/NumPy import.

The independent POSIX fault test wrote the first member, simulated failure
before the result member, closed the descriptor, and observed:

```text
public-result/
└── universal_selector_packet.bin
```

The public directory and partial member persisted. A lock/root/import/GPU
failure can similarly leave an empty public directory, while a failure between
the two writes leaves the selector alone.

Required repair: write to a hidden create-new staging directory, fsync and
enumerate all expected members, rehash them from held descriptors, write a
completion index last, reverify staging, and atomically publish once with
no-replace semantics. Constructor and pre-publication faults must clean or
quarantine staging; no failure may expose a complete-looking final tree.

### 5. The executable is not universal across SwiGLU-MoE shapes

The design text claims universal canonical SwiGLU-MoE triplets and shape-based
zero padding. The executable instead hard-codes:

```text
experts = 6
rows = 768
columns = 2048
records = 18
shape = [768, 2048]
```

`_validate_file_descriptor` rejects any other shape. There is no valid-tail
field, block padding implementation, shape-derived stream count, unequal-role
geometry, expert-count path, or tail rate/page ledger. The one accepted shape
is exactly divisible by both 4,096 and 262,144, so no tail is exercised at all.

The conditional *rule* is potentially shape-portable over 4,096-value blocks,
but the frozen experiment and physical format are not. Required repair: either
state honestly that this is a Qwen-geometry-only pilot while making no
universal-codec claim, or add a canonical arbitrary-shape adapter and prove for
every supported tail/layout that physical rate remains at most 2.5 bpw and
each selected expert remains below 2x routed reads.

### 6. Runtime and review authority are not authenticated

Neither executable has a runtime-environment lock. The reports record Python,
NumPy/CuPy versions and a device name only; no held interpreter hash,
distribution RECORD/tree digest, CUDA runtime/driver binding, device UUID/PCI
identity, or dependency graph is verified. Stage 0 opens and copies the source
panel's payload bytes before importing NumPy and CuPy, so dependency failure or
substitution occurs after sensitive inputs are already open.

The two authorization values are public string constants in the producer
source. No signed or externally pinned independent-review receipt is required.
They are useful typo guardrails, not authority. The statement
`source_package_authorizes_gpu: false` is documentary and is not enforced by
the executable once the published string is supplied.

Required repair: bind a reviewed runtime snapshot and device identity before
any payload opens or numeric import, and require an externally pinned review
receipt tied to the exact source root and permitted action. This BLOCK report
issues no such receipt.

### 7. The 73/72 read result is a layout calculation, not a decode trace

The current formula is correct for the declared aligned Qwen-sized layout.
However, stage 0 never decodes any opaque coarse stream and never emits or
reads an expert frame. It separately hashes reservoirs, then consumes free
evaluation reconstruction/symbol files. The constant `73/72` is copied into
the JSON report; there is no unique-page trace, owner map, compressed-read
counter, repeated-pass counter, or physical container parser.

This does not invalidate the arithmetic. It means the result remains a planned
upper-bound layout, not measured MoE inference traffic. A finite successor
must independently decode one chosen expert from the actual container and
report unique 4-KiB pages plus repeated compressed/HBM reads. A second pass is
not automatically forbidden, but it must be counted.

## Additional findings

- The documented complete-object early stop is not implemented; all matrices
  are evaluated. This is an efficiency defect, not a mathematical false pass.
- The jackknife path does not guard every leave-one-expert denominator, and
  role capture divides without first requiring positive role energy. A hostile
  zero-energy fold can crash rather than produce a fail-closed result.
- Panel IDs need not be nonempty and descriptor digest syntax checks lowercase
  length but not hexadecimal syntax. Later exact hash comparisons catch a bad
  digest, but the lock parser should still be canonical.
- The producer manifest correctly lists its seven non-manifest members, and
  the verifier catches mutation/extra members under the documented invocation.
  The problem is its execution boundary, not its arithmetic inventory.
- The continuous oracle legitimately reads the residual at the evaluator; it
  is explicitly source-leaking and not an emitted decoder. The invalid free
  channel is the unproved relationship between coarse bytes and the alleged
  decoder-visible reconstruction/symbol records.

## Boundary decision

This revision contains a useful, mathematically sound **continuous
rank-384 conditional-frame oracle** and exact arithmetic for one Qwen-sized
layout. Preserve those ideas. It does not yet contain a frozen 384-bit finite
code, authenticated source/runtime launch chain, coarse-to-decoded binding,
complete-or-absent result publication, universal tail adapter, or measured
MoE read trace.

The next revision should split the work cleanly:

1. a repaired authenticated continuous pilot may test the exact fixed Qwen
   geometry after an independently sealed N18 coarse producer exists; and
2. a separately frozen finite DH384 backend must define the literal QC/trellis,
   rational scale, global bytes, fine records, independent decode and physical
   container before it can be used in TACTIC-CAGE.

Until both boundaries are independently audited, do not launch CuPy or open an
actual coarse/Qwen payload from this producer root. The current source verdict
is **BLOCK**.
