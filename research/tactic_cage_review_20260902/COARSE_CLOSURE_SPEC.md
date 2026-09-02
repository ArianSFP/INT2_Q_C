# Real `307/128`-bpw N18 coarse closure

Date: 2026-09-02

## Verdict

`tactic_actual_coarse_n18_v3` is a byte-layout and security scaffold, not a
codec.  It has no packet language, numerical encoder/decoder, independently
reconstructible 1,228-byte microblock, or authenticated residual/fine handoff.
Its POSIX suite is also not green.  Preserve v3 and build a new sibling.

No eligible lower-rate artifact currently exists:

- v1 has an executable numerical wrapper and builder, but no lower-rate
  artifact and several obsolete loader/header/termination assumptions;
- v2 has the strongest packet grammar, geometry, seed and canonical
  re-encode rules, but no numerical bridge;
- v3 only proves byte identities;
- the old 2.5-bpw STRATA/POLAR artifact has the wrong topology and cannot be
  relabelled as N18;
- the conditional-dyadic package explicitly has no authenticated `307/128`
  coarse object;
- the VORPAL selection is about 2.48595 bpw, uses a different topology and
  leaves only about 57.55 bits per 4,096 values, not the required 416
  fine-plus-metadata bits.

The minimum safe path preserves complete 78,592-byte N18 packets, combines
v2's packet/source primitives with v1's numerical bridge, and initially makes
only whole-N18 role geometries target-eligible.  A real arbitrary-shape
`<=2.5`-bpw target-eligible microblock/tail codec is a separate quantizer
project.  The padded whole-N18 compatibility mapping is already universal over
accepted finite BF16 shapes, but nondivisible shapes may fall outside target.

Specification status: `BLOCKED_BEFORE_SOURCE_IMPLEMENTATION`.  The exact
normal/escape grammar below is frozen, but the composite selector/QC/global
subgrammars remain explicit design obligations rather than claimed bytes.

## Whole-N18 packet

Use magic `TACN18C4`, version `4`, with this exact 128-byte little-endian
header:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | magic |
| 8 | 2 | version |
| 10 | 2 | header bytes = 128 |
| 12 | 4 | `N = 262144` |
| 16 | 1 | profile `q = 164` |
| 17 | 1 | role ordinal |
| 18 | 2 | flags: normal `0x000F`; zero escape `0x8000` |
| 20 | 4 | canonical rows |
| 24 | 4 | canonical columns |
| 28 | 4 | tile ordinal |
| 32 | 4 | valid values |
| 36 | 4 | SC seed |
| 40 | 8 | RHT seed |
| 48 | 4 | transmitted positive FP32 scale |
| 52 | 4 | logical bit length |
| 56 | 32 | payload SHA-256 |
| 88 | 16 | algorithm ID |
| 104 | 20 | reserved zero |
| 124 | 4 | CRC32 over bytes 0--123 |

All integer fields are unsigned little-endian; scale is IEEE-754 binary32.
Role ordinals are `0=Gate`, `1=Up`, `2=Down-transposed`.  CRC is zlib/IEEE
CRC-32 over header bytes `[0,124)`, stored as little-endian u32.

In normal mode, let `A` be the number of canonical arithmetic-code bits and
freeze:

```text
logical_bits = A + 32
payload_bits = arithmetic_bits || 0^32
A <= 627,680
```

The guard is appended at bit granularity, before terminal byte padding.  The
payload occupies `[128,128+ceil(logical_bits/8))`; its SHA-256 covers exactly
those bytes.  Terminal byte padding and all remaining reservoir bytes through
offset 78,591 are zero.  The usable arithmetic capacity is 627,680 bits, only
992 bits above the 626,688-bit nominal profile.  The decoder may consume the
physical guard but never read beyond `logical_bits`.  Canonical re-encode must
reproduce arithmetic bits, the exact guard, terminal padding and the whole
packet.  Before implementation, prove by a frozen arithmetic-state bound or
exhaustive implementation argument that 32 guard bits cover every legal decode
path; otherwise increase the charged guard and reduce `A` accordingly.

Reject trailing data, nonzero fill, repaired-CRC tampering, wrong seeds, digest
mismatch and any read beyond the declared logical end.

Zero-escape mode is canonical:

```text
flags = 0x8000; all other flag bits zero
logical_bits = 0
payload_sha256 = SHA256(empty)
scale = canonical IEEE binary32 1.0
all payload/fill bytes = zero
```

Role, shape, tile, valid count and seeds remain normally bound.  The decoder
emits `valid_values` zeros and discards the padded suffix.  The guard applies
only to normal mode.  Escape parsing and byte-reencode are separate canonical
paths.  Without escape, q=164 overflow leaves a partial encoder and violates
universal mapping; target eligibility requires no escape packet.

The encoder attempts normal mode exactly once with the frozen profile and
seeds.  It emits normal mode iff the rounded scale is finite and positive,
`A<=627,680`, all decoded canonical symbols fit I16, and independent
decode/re-encode invariants pass.  An exactly zero padded vector, arithmetic
capacity overflow, or mathematical I16 representability failure emits the
canonical zero escape.  Seed, profile or transform retry is forbidden.
Implementation disagreement, digest failure or noncanonical re-encode is a
terminal error, never an escape trigger.

The 16-byte algorithm ID is:

```text
SHA256(
  b"UNIPOLAR-N18-307-C4-ALGORITHM\0" ||
  bytes.fromhex("85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e") ||
  bytes.fromhex("062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0")
)[:16]
```

`1,228*64=78,592` is only a byte identity.  The 128-byte header is not 64
independent two-byte headers, and no independently decodable 1,228-byte packet
exists.  Do not expose the microblock as a codec unit.

## Canonical source mapping

1. Stored Gate and Up are `[intermediate, hidden]`.
2. Stored Down is `[hidden, intermediate]`; transpose to contiguous
   `[intermediate, hidden]`.
3. Interpret sources as little-endian BF16, require finite values, and flatten
   canonical matrices row-major.
4. Tile `flat[t*262144:(t+1)*262144]`; zero-pad only the last whole N18 packet
   and bind its true `valid_values`.
5. Role order is Gate, Up, transposed Down.
6. Seeds depend only on role, rows, columns and tile ordinal--never model,
   checkpoint, layer, expert or provenance identity.

The compatibility tail rule stores a fully padded N18 packet for every partial
tail.  It works for every positive shape but may exceed 2.5 bpw.  Initial target
eligibility therefore requires every role value count divisible by `2^18`, no
escape packet, actual `R in [2.15,2.5]`, and the required MSE.  V3's aggregate
`floor(307*tail_values/1024)` language has no quantizer, scale, termination or
decoder and is retired.

Compatibility tails are coarse-only.  The 48-byte-per-4,096-value fine
composite has no charged partial-tail fine language and therefore makes no
arbitrary-shape composite claim.  For the frozen divisible panel, the exact
claims are coarse `R=307/128`, composite `R=2.5`, no escape, and
`F=MSE*2^(2R)<=0.8`, hence `MSE<=0.025` at `R=2.5`.

Scale construction, including tails, is exact:

```text
BF16 -> FP32 -> FP64
append canonical FP64 zeros to length N
apply the frozen orthonormal forward transform
RMS over all N transformed coordinates
round once to transmitted IEEE binary32
```

Normal mode requires a finite positive scale.  An all-zero padded vector uses
zero escape.  Distortion and source-energy accounting exclude padded suffixes
and use only `valid_values`.

## Frozen numerical code

- signed RHT;
- one transmitted positive FP32 block-RMS scale;
- 64-point Q31 alphabet `0.25*[-31,...,32]`;
- six MAP-SC levels;
- profile `q=164`;
- nominal test-channel rate `1.75 + 164/256 = 2.390625` bpw;
- physical whole-packet rate `78,592*8/262,144 = 307/128 = 2.3984375` bpw;
- payload capacity 627,712 bits versus nominal 626,688 bits before the charged
  guard; usable arithmetic capacity 627,680 bits and headroom 992 bits after
  the charged guard;
- decoded index `p in [0,63]`, integer `j=p-31`;
- exact symbol `h=diag(signs)*H_N*j`, checked to fit signed I16;
- reconstruction `float32(h*(0.25*scale/512))`.

The independent decoder/re-encoder must regenerate every SC frequency and
decision, recover indices/I16 symbols/F32 reconstruction, re-encode arithmetic
symbols, and compare logical length, payload and the complete fixed packet
byte-for-byte.

The normative prototype sources are pinned, not discovered:

```text
independent decoder/auditor
85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e

POLARIS SC encoder
062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0
```

The successor design lock must extract and freeze from those exact bytes the
SC/RHT seed domains and packed preimages, `H_N diag(signs)/sqrt(N)` forward and
inverse order, SplitMix64 sign rule, BEC freeze construction and ties, q=164
capacities/test distortion/eta, MAP-SC ties, frequency conversion/clamping,
arithmetic state width/termination/bit order, and every FP32/FP64 rounding
point.  Normative synthetic golden packets must cover all six levels,
overflow/escape, maximum logical length and partial tails.  An independent
decoder is a separate implementation checked against this frozen grammar and
the goldens; it may not call the encoder's decode path.

## Exact six-expert layout

For one complete 4,096-value planning block:

```text
total       1,280 bytes = 10,240 bits = 2.5 bpw
coarse      1,228 bytes =  9,824 bits
metadata        4 bytes =     32 bits
fine           48 bytes =    384 bits = 0.09375 bpw
```

The four metadata bytes are an amortized equality for this exact six-expert
panel, not a universal literal per-block field.

| Unit | Coarse | Fine | Metadata equivalent | Total share |
|---|---:|---:|---:|---:|
| one N18 / 64 blocks | 78,592 | 3,072 | 256 | 81,920 |
| one role / 384 blocks | 471,552 | 18,432 | 1,536 | 491,520 |
| one expert / 1,152 blocks | 1,414,656 | 55,296 | 4,608 | 1,474,560 |
| six experts | 8,487,936 | 331,776 | 27,648 | 8,847,360 |

Freeze the container:

```text
global                     [0, 24,576)
expert e base              24,576 + e*1,470,464
expert header              [base, base+512)
18 coarse N18 packets      [base+512, base+1,415,168)
1,152 x 48-byte fine       [base+1,415,168, base+1,470,464)
```

Ordering is exact:

```text
coarse packets: role-major, then N18 tile ordinal
fine fields:    role-major, then 4096-value microblock ordinal
fine_index = role*384 + microblock
coarse_tile = floor(microblock/64)
position_within_N18 = microblock mod 64
```

The global subintervals are literal:

```text
[0,4096)       schema/layout
[4096,20480)   selector
[20480,22528)  QC tables
[22528,24576)  seeds/fixtures/reserved
```

This allocation is not yet a finite composite grammar.  Before source
implementation, freeze a binary schema for every subinterval and the 512-byte
expert header: exact magic/version, field offsets/types, logical lengths,
expert count/order, frame offsets/counts, role shapes, algorithm identifiers,
selector/QC/seed payload definitions, per-region hashes, CRCs, canonical
padding and rejection rules.  Each selector/QC/reserved region needs its own
declared content and zero-fill policy.  The expert header must bind geometry,
fixed offsets/counts, algorithm versions and global/coarse/fine aggregate
digests.  Budgeting these fields without their content would repeat v3's
central defect; no finite composite claim is permitted until the companion
grammars and hostile tests exist.

## Residual provenance

For every fine block bind:

- held source byte size/SHA-256, stored shape/role and canonicalization;
- canonical flat start, valid count and N18 padding;
- coarse packet offset/length/SHA-256;
- decoded index-I16, canonical-symbol-I16 and reconstruction-F32 digests;
- residual exactly as
  `FP64(BF16-to-FP32 source)-FP64(decoded coarse F32)` on valid values;
- residual-record digest;
- 48-byte fine offset/SHA-256;
- final reconstruction digest and directly recomputed FP64 SSE/MSE.

The fine encoder may consume only the bound residual and decoder-visible
coarse state.  Decode may never read a source/reference tensor.

These detailed source/residual records are evaluation and audit sidecars; they
do not affect decoding and are not silently added to the codec rate.  Every
decoder-visible semantic must be inside the charged global/expert grammar.  If
the decoder needs a receipt digest, that digest occupies a charged field.
Source identity/provenance may authenticate scoring but never select
reconstruction behavior.

## Operational read proof

Instrument every compressed-file read with owner, phase, held FD identity,
offset, requested/returned bytes, buffer digest and touched pages.  Emit the
ordered trace, requested-byte sum, coalesced interval union, distinct pages,
maximum interval multiplicity/pass count, resident bytes between coarse/fine,
and separate H2D/D2H provenance.

For the frozen panel, read the six-page global packet once and the selected
359-page expert frame once, then keep the frame resident:

```text
requested = unique = 24,576 + 1,470,464 = 1,495,040 bytes
owner physical share = 1,474,560 bytes
amplification = 73/72
```

A second expert-frame scan is `(6+2*359)/360 = 2.011111...` and fails.  A
unique-page union without ordered request evidence is insufficient.

This is an application-requested compressed-byte and touched-4-KiB-page metric,
not a claim about storage-device, filesystem-cache or PCIe traffic.  Freeze the
denominator as total physical panel bytes divided by six:
`8,847,360/6=1,474,560`; it exceeds the selected 1,470,464-byte frame because
it includes the owner's equal share of the global packet.  H2D/D2H remains a
separate measured ledger.

## Smallest safe build sequence

1. Create a new sibling and freeze the whole-N18 packet, source mapping,
   normal/escape, composite container and read-trace grammars.  Leave v3
   untouched.
2. Implement standard-library parsers/accounting and hostile format tests,
   including guard, EOF, padding, escape and tails.
3. Implement immutable source/dependency loading, executable/runtime binding,
   numeric-module preload rejection, no-follow error normalization and
   descriptor stability.
4. Port v1's encoder and independently implement the decoder/re-encoder from
   the pinned cores, without claiming execution evidence yet.
5. Obtain an independent source inventory and full hostile POSIX audit.
6. Resolve and freeze the exhaustive runtime.
7. Run authenticated synthetic numeric encode/decode/re-encode, tail, escape,
   overflow, CUDA/telemetry and one-read `73/72` tests.
8. Only then build and independently audit the real coarse object.  Require
   every packet to parse, decode, hard-EOF, byte-reencode and avoid escape.
9. Only after a coarse survivor implement the residual/fine/composite producer.
   Treat an arbitrary-shape exact-rate microblock/tail codec as a later new
   quantizer project, not a documentation patch.

## Evidence

- `research/tactic_actual_coarse_n18_v1/`
- `research/tactic_actual_coarse_n18_v2/`
- `research/tactic_actual_coarse_n18_v3/`
- `research/tactic_actual_coarse_n18_v3_independent_audit_20260902/`
- `research/tactic_conditional_dyadic_coset_v2/`
