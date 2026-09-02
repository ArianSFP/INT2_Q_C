# LOGIC-Q label-flexible algebraic gate v0

Date: 2026-09-02

Status: **frozen source-only mechanism and accounting gate**. This package was
built without opening, statting, hashing, enumerating, or otherwise resolving a
model checkpoint, Qwen tensor, current-codec payload, completed coarse result,
or matched-control artifact. It imports neither CuPy nor CUDA. An independent
source audit must bind the exact `SOURCE_MANIFEST.json` before anyone writes or
runs a payload adapter.

## The question

LOGIC-Q does not ask whether labels already chosen by a nearest-neighbour
quantizer are compressible. It jointly chooses legal four-level labels and a
small, finitely decodable algebraic description by minimizing

```text
sum_i weight_i * (source_i - reconstruction_i[label_i])**2
    + lambda * physical_packet_bits.
```

This is the correct early test for the proposed architectural discontinuity:
move labels near cell boundaries toward a short mathematical object, rather
than entropy-code one frozen label assignment.

The audited reference point motivating the gate is

```text
R = 2.5 bpw
D = 0.030902167403153148
F = D * 2**(2R) = 0.9888693569009007
```

and the final target is `F <= 0.8` at a physical rate in `[2.15, 2.5]`.
At unchanged distortion this needs `0.1528899669629145` physical bpw of real
saving; at 2.5 bpw it needs `19.0995257%` less MSE. A source-only fixture cannot
establish either result.

## Frozen finite families

Every family reconstructs literal numeric values from a byte string. A
64-byte header is paid once per role matrix. It contains the role, shape,
family selector, public profile selector, block size, family parameter, scale
payload length, and literal family-payload bit length. Every quantization block
pays one transmitted BF16 scale. The profile selector occupies a complete
header byte. The final payload is byte padded. An expert envelope pays another
64-byte header, aligns each of the three role packets to 64 bytes, and pads the
complete expert object to 4096 bytes. RM blocks are individually byte-aligned;
the GF(2) and ROMDD family payloads are byte-aligned once. Those zero bits are
inside the optimized physical objective and are verified by the decoder.

The implemented modes are:

1. **LITERAL4** — two bits per ordered four-level label. This is the finite
   comparison packet, not an entropy ideal.
2. **RM1+E** — one first-order Reed–Muller (affine Boolean-function) codeword
   per Gray bitplane and public power-of-two block, plus a jointly optimized
   exception overlay. Blocks with at most 4096 affine-codeword pairs are
   searched exactly under the four-level distortion; larger blocks use a
   bounded, explicitly non-exact list.
3. **GF2-RANK+E** — two raw GF(2) factor products, one per Gray bitplane, plus
   the same exception overlay. The implemented packet transmits literal
   `U,V` factors and therefore costs `2*r*(rows+cols)` bits before exceptions.
   It does **not** claim the much smaller ideal rank-matrix counting bound.
   Search is exact only for tiny enumerated fixtures; the scalable alternating
   factor search is a non-promoting heuristic.
4. **ROMDD+E** — a bounded reduced ordered mixed-radix decision diagram plus
   exceptions. Its coordinate domain factors the exact role matrix dimensions;
   it never pads a 768-row axis to 1024 or invents a fourth role.

For every evaluated base object the exception optimizer is exact: for each exception
count it selects the positions and alternative labels that minimize the true
weighted four-level distortion, then charges a fixed count, a combinatorial
subset rank, and a base-3 replacement rank. Decoder-invalid unused binary
rank values reject. The optimizer compares the resulting byte-aligned physical
payload length, not an unpadded ideal bit count.

## The rank-680 correction

For a `768 x 2048` role, an implemented raw-factor rank-680 packet costs

```text
2 * 680 * (768 + 2048) / (768 * 2048)
    = 2.4348958333333335 bpw
```

before the matrix header, scales, exception count, exceptions, alignment, and
expert framing. It therefore cannot inherit the proposed `1.8469 bpw` rate.
That lower number is approximately the *ideal enumerative* count of two exact
rank-680 binary matrices, `2*r*(rows+cols-r)` bits, and is recorded only as an
unimplemented optimistic bound. A future canonical rank-matrix serializer must
be implemented and independently decoded before it can replace the raw-factor
ledger.

Even finding a nearest low-GF(2)-rank matrix under ordinary Hamming loss is
NP-complete already at rank one in the cited binary-matrix literature. The
joint four-level, weighted-distortion, rank-680 problem is not made exact by an
alternating factorization. A miss from the bounded heuristic cannot hard-kill
the global GF(2) family.

## Exact coordinate-domain rule

Three canonical role packets are separate. For the aligned `768 x 2048`
geometry, each role uses the exact mixed-radix domain

```text
3 x 2 x ... x 2   (3 followed by nineteen binary digits)
```

with `3 * 2**19 = 1,572,864` sites. The expert has exactly three such packets.
A naive 23-bit truth table has `2**23 = 8,388,608` sites but only `4,718,592`
valid role/row/column tuples. The other `3,670,016` values are not free don't-
care entries; this package never creates or silently masks them.

## Hard-kill ordering

Before any expensive search, the gate computes for each mode:

1. mandatory physical bytes with zero exceptions;
2. whether those bytes already exceed 2.5 bpw;
3. the largest descriptor/rank allowed by the unchanged-MSE saving budget;
4. an optimistic `F` lower bound combining nearest-label distortion with the
   mode's impossible-best minimum packet rate, padded to at least 2.15 bpw.

If even that optimistic lower bound exceeds `0.8`, the finite family is safely
killed. Exact small-block RM and exact tiny-GF2 enumeration can additionally
hard-kill their *enumerated* code families. A miss from RM list search, GF(2)
alternation, or bounded ROMDD depth is only a bounded negative result.

## Held-out and matched-Gaussian protocol

A future production panel must contain exactly one Gate, Up, and canonical
Down-transposed component for every layer/expert slot, with one identical slot
universe in every layer. At least ten whole layers and five whole-test-layer
clusters are required.

- Frozen hash-ranked whole layers form the untouched outer test.
- Among remaining layers, frozen whole expert slots form validation; those
  slots never enter training.
- Global block size, lambda grid, RM list width, GF(2) rank bank, ROMDD digit
  order/depth, and any stopping rule are selected on train/validation only.
- A test component may choose among modes only when its selector is literally
  present and charged in its packet.
- Uncertainty is a paired whole-test-layer cluster bootstrap, never a
  per-weight IID interval.

Eight frozen matched-Gaussian seeds are applied to the immutable raw panel.
Every public quantization block is independently generated to match that
source block's binary64 mean and centered energy, then the complete profile
selection, BF16 scale fitting, label-flexible search, exception packing,
byte/page framing, decode, and model selection are rerun. Controls never accept
prebuilt labels and cannot create a source pass. The source must clear the
absolute operational gate first.

## Source-only commands

These commands touch only this package and deterministic synthetic arrays:

```bash
python -I -B research/logic_q_label_flexible_algebraic_gate_v0/test_source_only.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v0/run_source_free_fixture.py
python -I -B research/logic_q_label_flexible_algebraic_gate_v0/verify_source.py \
  --package research/logic_q_label_flexible_algebraic_gate_v0
```

## Claim boundary

The package proves only that the finite packet definitions, weighted searches,
rate bounds, exact-domain accounting, synthetic controls, and decoders agree.
It provides no model result, no claim that algebraic structure exists in real
weights, no complete 2.15–2.5-bpw codec, no `F <= 0.8` result, no below-2x
routed-read result, and no universality result. A Qwen survivor would still
need transfer to a disjoint SwiGLU-MoE family under the frozen protocol.

See `PRIOR_ART.md` for the research boundary and primary references.
