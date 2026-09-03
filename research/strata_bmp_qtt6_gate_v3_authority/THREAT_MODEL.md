# V3 authority threat model

The adversary may choose paths, JSON text, digests passed to the lower-level
evidence verifier, route names, capability names, metric values, and receipt
values. It may rename, hard-link, symlink, duplicate, mutate, truncate, extend,
or add files. It may label a layout calculation as a read measurement or reuse
model bytes as a matched control.

The trusted base for this source sibling is limited to its later externally
pinned source manifest, the four compiled v2/audit hashes, Python's standard
library, and a cooperative filesystem that does not defeat stable pre/post
metadata checks. This is not a hostile-kernel or remote-attestation boundary.

Production authorization requires a launch-manifest SHA-256 compiled into the
source before execution. This freeze deliberately has no such digest. The
lower-level verifier is available for independent audit and hostile tests, but
its caller-supplied digest is not a trust decision.

Every external capability is a flat, exact regular-file closure with a
canonical manifest. Execution and independent-audit receipts are separately
pinned and must bind the same implementation and capability ID. Producer,
executor, and auditor identities must be distinct. Production rejects fixture,
dummy, and self-authored evidence.

The authority reopens literal BF16 sources and STRATA packets, rejects path,
inode, and content aliases, binds adapter reconstruction to the independent
scorer, recomputes score formulae, and validates event-level page reads per
routed expert. It does not accept a semantic claim merely because a SHA-256
string is well formed.

Out of scope for this source freeze: payload availability, CUDA execution,
filesystem attacks by a privileged kernel, cryptographic identity signatures,
and the scientific truth of receipts that have not yet been produced. Those
are why production remains held.

