# Threat model and claim boundary

The producer, runtime snapshot, Wasm decoder, canonical encoder, evidence
bundle, receipts, and their filesystem paths are treated as untrusted inputs.
SHA-256 values are meaningful only when obtained from an independent trusted
channel. Canonical JSON, exact regular-file closure, non-link path traversal,
pre/post read identity, runtime import closure, and guest capability checks are
reviewed as fail-closed mechanisms.

The Python host and operating system are within the trusted computing base.
This review specifically notes that v4 records but does not reobserve the
current Python ABI/platform/target, and that the loaded-native check does not
close over libraries outside the copied runtime snapshot.

The separate runtime and semantic audits are assumed trustworthy only when
their expected pins originate outside the evidence producer. Self-authored
PASS JSON plus self-selected expected hashes is not independence.

The semantic decoder is adversarial until the external semantic audit proves
otherwise. Zero imports prevent the canonical encoder from directly reading a
packet; they do not prevent the decoder from embedding packet bytes in the
opaque semantic-state buffer.

The reviewed read ledger is the declared page-rounded callback supply metric.
It is not a measurement of all parent/child copies, decoder memory traffic, or
GEMM bandwidth.

No network, model checkpoint, Qwen tensor, compressed packet, matched control,
or scientific evidence was accessed. No runtime, F-score, rate-distortion,
cross-model portability, or production authorization is claimed.

