# Threat model and claim boundary

The mutable producer trees, their filesystem paths, coarse worker/auditor
closures, capabilities, receipts, and callers are treated as untrusted. Hashes
confer authority only when their expected values originate from a separate
trusted publication.

The atomic bootstrap is part of the trusted computing base. Its in-script
self-hash runs after Python has already parsed and begun executing it, so an
external pre-execution verifier or immutable host image is required against a
hostile filesystem. Isolated Python startup is also required to prevent module
shadowing before that check.

Once authenticated, the snapshot mapping contains immutable `bytes` values and
the project modules are compiled from those values. Live NumPy/CuPy and Python
standard-library dependencies are outside this source snapshot.

The coarse program is untrusted canonical JSON data. The fixed interpreter
supports only a zero-output opcode and has no program-level path, callback,
import, decoder-object, or network operation. The host authenticator itself
accepts filesystem paths for the external evidence objects.

Worker/auditor independence is not inferred from self-selected hash values.
The review requires externally controlled, non-aliased audit roots for any
future payload use.

No network, model tensor, coarse artifact, source-free numerical fixture,
CuPy runtime, physical compression result, Qwen result, or RD claim was used.

