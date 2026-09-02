# Independent audit threat model

The externally pinned object is the frozen source-only producer package, not a
model checkpoint or codec payload. The audit authenticates its exact manifest,
member hashes, source-root hash, regular-file closure and absence of symlinks
before importing producer code.

Protected properties:

1. The v0 semantic collision repairs reject GF(2) matrix and QTT rank/gauge
   aliases at the serialized representation boundary.
2. All six uint16 geometry fields fail with `CodecError` before `struct.pack`
   when out of range.
3. Packet-rate arithmetic is literal and complete-rate bounds use exact integer
   arithmetic.
4. A source-only fixture, static test or CuPy receipt cannot be promoted into a
   Qwen, physical-composite, F-score or routed-read claim.
5. The CuPy receipt must come from a distinct `-I -B` process, bind the pinned
   source root, identify the installed CuPy distribution and active device,
   expose dedicated-pool samples, and reproduce an independently calculated
   rank-0/rank-1 winner packet.

The suite intentionally reproduces two non-adversarial producer defects rather
than hiding them: a valid skew-geometry packet exceeds the fixed 2,048-byte
candidate-workspace slot, and the frozen source verifier is blocked both by an
unsupported README CLI option and by its own member-order predicate. A passing
audit receipt therefore confirms the findings; it does not waive them.

The production-hook check is treated only as a syntactic fail-closed gate.
Digest strings are not dereferenced by this audit. Production must separately
authenticate every referenced object, verify the source/model manifest, repeat
complete codec selection on controls, score original BF16 weights and execute
the routed-read ledger.

Python, NumPy, CuPy, CUDA, filesystem and host compromise are outside this
source audit. SHA-256 collision resistance is assumed. Dedicated CuPy pool
bytes are not whole-process RSS. Runtime receipts are execution pending until
actually produced and immutably hashed; source files alone are never a PASS.
