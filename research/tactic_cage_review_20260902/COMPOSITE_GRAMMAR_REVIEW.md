# TACTIC-CAGE composite grammar review

Date: 2026-09-02

Status: **PROPOSED ENVELOPE; BLOCKED BEFORE CONFORMANCE**

This review separates byte layouts and packets that already exist from fields
that are only design proposals.  It authorizes no compression or MSE claim.

## Bottom line

The physical composite envelope and active selector can be frozen.  The old
Q12/QC refinement cannot: there is no finite mixer, trellis, coefficient map,
or 2 KiB QC table in the reviewed sources.  The smallest evidence-backed
closure is therefore:

1. retain the literal TACTIC-DH384 v2 ordinal-17 selector;
2. explicitly use the finite v3.1 sign bridge as the candidate fine branch;
3. require the unused 2 KiB QC region to be all zero;
4. retain v2's shape/role/tile-only seed derivation;
5. add a versioned schema and extended expert header;
6. withhold conformance until the C4 coarse numerical rules and fine
   floating-point ABI have executable golden vectors.

The expert-private v4 selector remains a separate inactive option.  Activating
it would change the algorithm and universality boundary.

## Exact physical envelope

The proposed 8,847,360-byte container is:

```text
global                                      [0, 24,576)
  schema                                    [0, 4,096)
  selector                                  [4,096, 20,480)
  QC                                        [20,480, 22,528)
  seeds/fixtures                            [22,528, 24,576)

expert e base = 24,576 + e * 1,470,464
  header                                    [base, base + 512)
  coarse                                    [base + 512, base + 1,415,168)
  fine                                      [base + 1,415,168, base + 1,470,464)
```

For role `r` and coarse tile or fine microblock `t`:

```text
coarse_offset = base + 512 + 78,592 * (6*r + t),       t=0..5
fine_offset   = base + 1,415,168 + 48 * (384*r + t),   t=0..383
```

This yields per expert:

```text
512-byte header
18 * 78,592-byte coarse packets = 1,414,656 bytes
1,152 * 48-byte fine records    =    55,296 bytes
frame total                     = 1,470,464 bytes
```

The layout identity is not evidence that either numerical codec works.

This is the self-describing archival envelope.  A lean fixed-decoder profile
may omit the deterministic selector, empty QC and seed-fixture regions and
bind their algorithms through the schema ID.  That removes 20,480 redundant
per-model bytes; exact accounting is recorded in
`INFORMATION_ACCOUNTING_CORRECTIONS.md`.  It does not remove any
source-dependent model or header bytes.

## Selector: evidenced exact packet

Use the literal v2 packet, not the incompatible v3 placement:

```text
relative [0,192)       exact canonical JSON plus LF
relative [192,3264)    12 * 256 u8 operation table
relative [3264,16384)  zero
```

The canonical JSON is:

```json
{"format":"TACTIC-DH384-UNIVERSAL-SELECTOR-v2","generator":"SplitMix64","generator_domain_u64_hex":"5441435449434448","ops":"swap/sign0/sign1","stages":12,"states":256,"universal_ordinal":17}
```

It is followed by one LF.  Exact constants:

```text
table SHA-256   e2bbe50df62c74e144f87e6894009d2745e10e0bda049b046c4f7b3567d81e14
table CRC32     047260cc
packet SHA-256  0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad
packet CRC32    70f6557e
```

Every table byte must be in `[0,7]`.  The decoder must regenerate the table
from the frozen SplitMix64 rule and reject a byte, hash, CRC, or zero-padding
mismatch.  The v3 selector has the same table at a different offset and must
not be accepted as an alias.

## QC: old branch absent; candidate empty grammar

The old Q12/QC branch is unimplemented and cannot claim these bytes.  If the
v3.1 sign bridge is explicitly selected, it needs no global QC table, so the
candidate subgrammar is:

```text
algorithm label  EMPTY_QC_SIGN_BRIDGE_V1
logical bytes    0
physical bytes   2,048, all zero
SHA-256          e5a00aa9991ac8a5ee3109844d84a55583bd20572ad3ffcd42792f3c36b183ad
CRC32            f1e8ba9e
```

This is a replacement design decision, not evidence that the missing Q12/QC
trellis exists.

## Seed rule: exact proposal awaiting normative adoption

The least novel rule is the N18 v2 carry-forward:

```text
suffix = struct.pack("<BIII", role, rows, columns, tile_ordinal)
SC  = little_u32(SHA256(b"UNIPOLAR-N18-307-SC-v2\0"  || suffix)[0:4])
SC  = 1 if SC == 0 else SC
RHT = little_u64(SHA256(b"UNIPOLAR-N18-307-RHT-v2\0" || suffix)[0:8])
```

Expert, model, layer and checkpoint identity are forbidden inputs.  The
proposed 2,048-byte seed packet has 736 logical bytes and zero fill, with:

```text
SHA-256 c6e398fce8c49a617d1c8ed7f1da2346592df6a300b501af6b6bf694c8c7d13b
CRC32   85d915ca
```

These values are reproducible but not normative until the C4 design lock
explicitly adopts the carried-forward domains.

## Schema and expert header

The proposed schema magic is `TCAGEC01`, version 1, occupying one 4 KiB page.
It binds:

- total/global/frame sizes and absolute offsets;
- the 18 coarse packets and 1,152 fine records per expert;
- canonical Gate, Up and Down-transposed role geometry;
- selector, QC and seed region hashes, CRCs and logical lengths;
- 16-byte coarse, fine and selector algorithm IDs;
- zero padding and a final CRC32 over the schema page prefix.

The proposed 512-byte expert header magic is `TCEX3841`, version 1.  It binds
the structural expert ordinal, public dimensions, three binary16 role gammas,
global/coarse/fine region hashes, algorithm IDs, frame offsets/counts and a
final CRC32.  Expert ordinal validates placement only; it must not affect a
seed, selector, reconstruction, or other numerical decision.

The existing C4 coarse algorithm ID is:

```text
0177ca52aaf9ae6e9432d1f89b6ffba7
```

Candidate selector and sign-bridge IDs were derived from the exact reviewed
source and packet hashes, but remain non-normative until the numerical ABI is
closed:

```text
selector candidate  8bc227c187b7b69d200e99b5edce0501
fine candidate      2ad21d48029974cc8676343bea7ee470
```

## One-pass routed read contract

The application makes exactly two compressed-file range requests:

```text
[0, 24,576)                         global pages
[base, base + 1,470,464)            one complete expert frame
```

The expert header is parsed from the resident second buffer.  Coarse and fine
may be rescanned only in resident memory.  The exact planned receipt is:

```text
requested = returned = unique = 1,495,040 bytes
touched pages = 6 + 359 = 365
owner share = 1,474,560 bytes = 360 pages
read ratio = 365/360 = 73/72
```

A short read, retry, overlapping request, second frame read, or another expert
read fails.  Reading the frame twice would be `(6 + 2*359)/360 = 2.01111...`
and violate the strict bandwidth requirement.

## Mandatory rejection surface

An independent decoder must reject at least:

- wrong file length, offset overflow, overlap, gap or bad region boundary;
- unknown magic, version, flags or algorithm ID;
- hash, CRC, logical-length or nonzero-padding disagreement;
- any selector byte differing from the literal v2 packet or regeneration;
- role/order/shape/count disagreement, including Down orientation;
- noncanonical, negative, nonfinite or negative-zero gamma;
- a C4 packet whose public role/shape/tile/valid-count binding is wrong;
- C4 digest, seed, guard, terminal-padding, re-encode or algorithm-ID failure;
- fine unpack/repack mismatch or nonfinite reconstruction;
- a second frame read or access to another expert;
- any model/checkpoint/layer/provenance/router/activation/source reference used
  as a decoder input.

Serialized source-fitted parameters inside the authenticated packet are legal
when every byte/page is charged.  The rejection above concerns an external
source reference or uncharged identity-dependent lookup, not ordinary PTQ
encoder adaptation represented in the literal message.

Canonical C4 escape packets remain part of the universal packet language, but
an evaluation artifact containing one is ineligible for the 2.5-bpw target.

## Hard blockers

1. The advertised Q12/QC fine branch has no executable definition.
2. The sign-bridge alternative must freeze direct canonical-symbol-I16 input,
   positive-zero gamma, exact binary64 accumulation/square-root order, every
   binary32 rounding point, FMA/contraction behavior, final addition and output
   dtype.
3. C4 still lacks frozen arithmetic termination, frequency conversion, MAP-SC
   tie rules, transform order and a guard sufficiency proof.
4. Normative coarse/fine golden packets do not yet cover all levels, capacity,
   escape/overflow and hostile tampering.
5. Until fine and QC modes are chosen, final schema/global/header hashes cannot
   be constants.
6. There is no actual independently decoded composite container or ordered
   range-read trace.  The 2.5-bpw and `73/72` values are layout identities only.

## Promotion boundary

No CAGE experiment may begin from a synthetic or relabelled coarse stream.
First emit and independently decode the actual `307/128`-bpw C4 coarse packet.
Only then may the same literal coarse word define a graph, posterior model,
adaptive tree or syndrome.  All promotion requires one nested physical byte
ledger, one source-domain reconstruction, and a literal one-pass receipt.
