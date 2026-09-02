# Independent v2 audit threat model

The audit authenticates one externally pinned source tree before producer
import. It treats source verification, CPU execution, GPU execution and payload
authorization as separate capabilities.

Protected properties:

1. Canonical manifest order, member hashes, source-root hash and exact regular
   filesystem closure must agree.
2. The source verifier's documented arguments must match its parser and its
   self-replay invocation.
3. Conservative logical capacity must never be represented as observed runtime
   allocation. Instrumented host ownership and measured CuPy-pool bytes must
   remain allocator-specific.
4. GF(2), QTT and ROBDD functions must retain unique accepted serialized
   semantics; six completed planes alone form the decoded 0..63 label.
5. A caller-generated digest or self-attested JSON receipt must not be mistaken
   for external reviewer trust or production evidence.

The independent suite deliberately constructs otherwise valid dummy artifacts,
controls, read ledgers and audit receipts. If the hook returns `authorized`, the
test records that as the expected B2 trust-boundary exposure. It does not bypass
the package's explicit production hold.

Python, NumPy, CuPy, CUDA, filesystem and host compromise are outside this
source audit. SHA-256 collision resistance is assumed. Pre/post metadata cannot
exclude an adversary capable of same-inode, same-size, same-timestamp mutation.
A CuPy memory-pool receipt is not process RSS. Runtime and payload claims remain
false until separately executed, reviewed and externally pinned.

