# Independent hostile audit: SILT-INT2 source-free mechanism v0

Date: 2026-09-02  
Producer under audit: `research/silt_int2_source_free_mechanism_v0`  
Verdict: **BLOCK** for sealing, payload access, or any universal-codec promotion.

This was a source-free audit. I inspected only the eight producer files and the
frozen universal contract. I did not discover, open, stat, hash, enumerate, or
copy any Qwen/model, current-container, or matched-control payload. I did not
edit or freeze the producer. The hashes embedded in the hostile test are an
audit-time observation, not a producer manifest or seal.

The transform algebra is sound and the producer is unusually clear about the
limited meaning of its synthetic fixture. The block is instead caused by three
concrete format/ledger defects plus missing execution authority: the unequal-
expert cold denominator is wrong, the nominally positive/universal geometry is
not safely bounded (and the header stops at 249 experts), and the public decode
API accepts noncanonical arithmetic packets that only the optional re-encode
path rejects. The mandatory GPU claim also has no reproducible exact-source
RunPod root yet.

## What was actually tested

`hostile_static_and_math_tests.ps1` ran locally and completed 1,656 independent
source/math checks. Its SHA-256 is
`b4714ca9037a148f553c160cf733e5eb196c62fa30ebefbd6f682e861416dced`.
The emitted receipt is `AUDIT_TEST_RECEIPT.json`.

The local host has no `python`, `python3`, or `py` executable and no CUDA/CuPy
runtime. The producer directory is also untracked and unsealed, so there is no
public commit or authenticated archive from which RunPod could obtain the exact
audited bytes. In accordance with the task boundary, I did not privately copy
the directory to RunPod. Consequently, producer-Python and real-CuPy execution
remain unresolved; the README statement that it was exercised on a 5090 is not
an independently auditable result receipt.

Logic-level results:

| Area | Result | Qualification |
|---|---:|---|
| GF(2) and Z4 pair lifting | PASS | Exhaustive over every input pair and all 8 selector IDs; inverse exact. |
| Recursive tree and odd carries | PASS | Independent roundtrips for alphabets 2/4, lanes 1, 2, 3, 5, 17, 97, 257, identity/reverse permutations, and all selector IDs. |
| Factoradic rank law | PASS | 300 independent rank/unrank trials for lengths 1..12 plus the 5!/6! byte boundary. Producer length/range checks are canonical for bounded inputs. |
| Q16 largest remainder | PASS | 200 independent rows; positive `uint16`-representable entries and exact sum 65,536. Producer/independent serialization layouts agree statically. |
| Page and total-byte charging | PASS | Canonical builder charges headers, metadata, model, payload, and zero padding through literal container length. |
| Per-expert ownership/cold ledger | **FAIL** | Uses `T/E`, not the byte-conserving owner share `G/E + F_e`. |
| Finite arithmetic grammar | **FAIL** | Canonical producer/re-encode is deterministic, but ordinary decode ignores `meaningful_bits` and zero-extends forever. |
| Arbitrary shape/expert support | **FAIL** | 249-expert directory ceiling; no pre-factorial/pre-allocation resource caps. |
| Independent decoder | PARTIAL | Separate implementation and exact canonical re-encode are valuable; parser repeats resource/canonicality weaknesses. |
| CuPy implementation | PARTIAL | Genuine device transforms and fail-closed import; no authenticated GPU replay or complete transfer/memory ledger. |
| Synthetic structured/control symmetry | PASS | Both classes get the same candidate space and separate train/search/model fit. It remains a constructed mechanism fixture only. |
| Universal SwiGLU-MoE codec contract | NOT REACHED | No Gate/Up/Down adapter, float-to-label quantizer, reconstruction/MSE score, 2.15--2.5 bpw packet, or held-out family test exists here. |

## Blocking counterexamples

### B1. Unequal expert frames can pass the reported cold gate and fail ownership accounting

The producer and independent receipt both calculate

```text
reported_amp(e) = (G + F_e) / (T/E).
```

For a literal expert-owned layout, storage ownership is

```text
owner_share(e) = G/E + F_e
cold_amp(e)    = (G + F_e) / owner_share(e).
```

The owner shares sum exactly to `T`; `T/E` does not describe expert `e` when
frames differ. Here is a counterexample consisting entirely of legal page
sizes:

```text
E = 8
G = 8192
F = [4096, 8192, 8192, 8192, 8192, 8192, 8192, 8192]
T = 69632
```

The current ledger reports a maximum of
`16384 / 8704 = 1.8823529411764706`, so it passes. Expert 0's correct
owner-aware amplification is
`12288 / (1024 + 4096) = 2.4`, so the same layout fails. Equal source weight
counts do not imply equal compressed frame sizes. Padding or inflating other
experts can therefore improve an unrelated expert's current denominator.

This is also an anti-padding-gaming failure. The physical rate correctly
charges the inflated bytes, but the routed-read gate can be made easier for a
small expert by adding bytes owned by other experts. The repair must emit an
exact byte-ownership decomposition, prove that its rational allocations sum to
the literal container size, and use that same decomposition in every cold
denominator. An instrumented page union should independently reproduce the
numerator.

### B2. The format is neither safely bounded nor fully positive-count portable

The 4,096-byte global header has a 104-byte fixed struct and 16-byte directory
rows. Its exact capacity is therefore

```text
floor((4096 - 104) / 16) = 249 experts.
```

Expert 250 is the first advertised positive count that cannot be emitted; 256
also fails. The design lock says only “positive lane/vector/expert counts” and
does not state this limit. A 128-expert container fits statically
(`104 + 128*16 = 2152` bytes), but no 128-expert runtime fixture exists.

More seriously, both parsers call `factorial(lanes)` to derive permutation
width before any explicit lane/resource maximum. The frame field permits a
32-bit lane count. A forged header can therefore request, for example,
`factorial(2^32-1)` before CRC/body rejection. Decode then allocates root and
detail arrays from unbounded `vectors` and `vectors*(lanes-1)`. The independent
container parser also starts unpacking `experts` directory entries without its
own prior `directory_end <= 4096` check.

Required repair:

1. publish exact format maxima, including an expert range that covers at least
   1..256 if that is the intended contract;
2. validate every scalar field and checked product before factorials, loops,
   slices, conversion, or allocation;
3. derive permutation-byte bounds without first materializing an adversarial
   factorial, or cap lanes before doing so;
4. cap decoded symbols, logical bytes, pages, and host/device allocation; and
5. make producer and independent parser reject with controlled format errors,
   not incidental `struct`, memory, or timeout failures.

### B3. `meaningful_bits` does not bound arithmetic decoding

Both decoders construct the arithmetic decoder with the payload bytes only.
Neither passes `meaningful_bits`; reads beyond the payload return zero forever.
The frame grammar permits `meaningful_bits == 0` and merely requires all bits
after it to be zero.

A one-byte all-zero arithmetic payload with `meaningful_bits=0`, a recomputed
frame body CRC, and a recomputed frame-header CRC therefore passes the ordinary
frame/container parser. `decode_container` can synthesize the declared number
of coefficients by infinite zero extension. The global header need not change
when frame length and directory remain unchanged. The resulting packet is not
the canonical finite encoding of those decoded coefficients.

`verify_decode_reencode` correctly rejects such a packet, but the normal
`decode_container` API does not invoke that gate. Thus the format has two
acceptance languages: “decodable” and “canonical.” That is unsafe for a finite
byte ledger and gives a malicious encoder a truncation surface.

The decoder must consume a bit-limited reader, reject required reads past the
declared meaningful termination, validate the termination rule, and enforce
canonical re-encoding (or an equivalent uniqueness proof) in the ordinary
parse/decode path.

## Secondary findings

### GF(2) selector aliases

The 3-bit tuple space contains eight IDs, but GF(2) has only six distinct maps:

```text
selector 2 == selector 7
selector 3 == selector 6
```

This is specific to GF(2); all eight are distinct over Z4 because negation is
not identity modulo 4. The current fixed three-bit representation charges every
ID fully, and independent re-encode preserves the literal ID, so the aliases do
not create a free-rate win by themselves. However, “canonical transform
metadata” is semantically false for GF(2), search counts redundant IDs, and the
decoder does not reject an alias in favor of a canonical representative. Freeze
one of two explicit policies: restrict GF(2) to six canonical maps, or state
that selector IDs are literal charged syntax with semantic aliases and never
count them as eight distinct transform choices.

### Root of trust and filesystem publication

The producer is untracked, has no manifest, and intentionally has no result
authority. `verify_source.py` loads `test_source_only` by ambient module name;
the runner imports three siblings before authenticating any bytes. Python site
startup/import state is not isolated, so an external result cannot establish
which code executed from its own receipt.

The output writer checks `exists()` and then reopens by pathname with
`write_text`/`write_bytes`. It does not hold a directory descriptor, reject
symlink/reparse traversal, use exclusive no-follow creation, fsync files and
the directory, or atomically publish an authenticated completion record. These
are result-authority blockers, even though the current outputs are labeled
unsealed.

### GPU and telemetry boundary

Static inspection confirms that every candidate performs real CuPy H2D,
modular tree kernels, and D2H; candidate ranking then deliberately uses the CPU
reference finite coder. There is no silent CPU fallback when the GPU path is
requested.

The receipt is nevertheless incomplete:

- no exact `h2d_bytes` or `d2h_bytes` is emitted;
- no peak host RSS is measured;
- sampled VRAM is total device-used memory, not baseline-subtracted run memory;
- the NVML sampler passes a CUDA logical device index directly to NVML, which
  can select the wrong physical GPU under `CUDA_VISIBLE_DEVICES`; and
- the default verifier may skip CuPy unless command-line flags are supplied.

For the canonical `2048/1024 x 97` search with eight candidates, source plus
metadata H2D is derivable as 305,832 bytes per search invocation:

```text
(2048+1024)*97                     source uint8
+ 8*(97*8 + 96)                   candidate int64/u8 metadata
+ (97*8 + 96)                     selected metadata replay
= 305832 bytes.
```

That number is absent from telemetry. Q16 model arrays are never sent to the
GPU because fitting/scoring is CPU-side; the receipt should say so explicitly
rather than leaving “model H2D” ambiguous. A repaired run should report exact
array/descriptor bytes by direction, baseline and peak host/VRAM, GPU UUID and
logical-to-physical mapping, driver/runtime/CuPy versions, and a measured
full-scale wall-time/peak receipt.

### Integrity is not result authority

Canonical zero bit/byte tails, page coverage, model SHA-256, and frame/header
CRCs are all checked. CRC32 is deliberately forgeable and is only an accidental
corruption check; it does not authenticate research provenance. The outer
sealed result must hash the complete container, source, verifier, decoder,
ledger, and telemetry receipt from an independent root.

## Scientific interpretation

The source/control protocol is symmetric enough for its stated purpose. Both
structured and iid-control classes receive the same eight-candidate search and
their own disjoint search-train, validation, model-fit, and evaluation data.
The hidden generating transform is intentionally one of the candidates for
both. This demonstrates that the mechanism can recover and code a dependency
constructed to match its state machine.

It is **only a mechanism proof**. The README and result schema say this
correctly. There is no evidence that real SwiGLU-MoE labels contain the same
checks, no Gaussian/structure-destroying payload result, no float reconstruction
or MSE, no physical bits per model weight in 2.15--2.5, and no Gate/Up/Down
shape/role adapter. A synthetic pass cannot be cited as source gain or as a
universal codec result.

## Pre-seal acceptance gates

Do not authorize payload access or seal v0. A repaired version should satisfy
all of the following before promotion:

1. Correct the owner-aware storage/cold ledger and add the unequal-page
   counterexample above as a mandatory test.
2. Define and enforce bounds before every factorial, product, directory loop,
   slice, and allocation; support and test 1, 128, 249, 250, and 256 experts
   according to an explicit format contract.
3. Make the ordinary decoder's accepted language exactly the canonical finite
   arithmetic language, including meaningful-length and EOF tests.
4. Freeze the GF(2) alias policy and test semantic/canonical behavior.
5. Publish the repaired exact source through a cryptographic, public/bootstrap-
   accessible root. An independent runner must authenticate bytes before
   import/execution and record the interpreter/environment closure.
6. Run the mandatory CuPy and 5090 suite from that exact root without a private
   workspace copy; capture exact transfer bytes, host/VRAM telemetry, device
   mapping, stdout/stderr, exit code, and artifact hashes.
7. Have an external auditor independently parse, decode, canonical-reencode,
   reproduce leaf digests, recompute physical/owner/cold ledgers, and verify
   source/control symmetry from the frozen result.
8. Only then build the separate universal SwiGLU adapter and run the contract's
   source/control, cross-family, physical-bpw, raw-MSE, and routed-read gates.

Until these are met, the correct status is **BLOCK — sound reversible algebra,
unsealed mechanism only, no payload or universal-codec authority**.
