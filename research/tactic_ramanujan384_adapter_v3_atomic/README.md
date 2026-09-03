# TACTIC Ramanujan-384 atomic authority v3

Date: 2026-09-03

Status: **frozen source architecture; runtime, CuPy, Qwen, and coarse-model
payload authority held**.

V3 is a narrow hardening layer over the frozen scalable-v2 mathematics. It
does not change the Ramanujan dictionary, Box--Muller controls, batched rank
search, packet format, rate ledger, or acceptance thresholds.

Two authority boundaries change:

1. The separately hash-pinned bootstrap reads every v3, v2, and v2-review
   member through a regular-file descriptor, verifies exact closure twice,
   constructs a private read-only snapshot, rehashes it, and only then compiles
   the runner from immutable in-memory snapshot bytes. Package code is never
   imported from the mutable producer tree.
2. The coarse decoder is an authenticated zero-import byte-buffer program, not
   a caller-provided Python object. Its source and auditor closures, capability,
   and exact-schema PASS receipt have distinct out-of-band hashes and roots.
   The tiny VM accepts only program bytes, coarse bytes, uint32 geometry, and
   the fixed role tuple. It has no path, callback, import, file, or network
   opcode. Every decoded role must match both the independent worker receipt
   and the independent source/reconstruction audit.

The included zero-output worker and periodic source are synthetic mechanism
fixtures. Their audit documents are co-generated fixture evidence, not a real
independent production audit. A separately controlled worker/auditor bundle is
mandatory before model payload use.

`test_source_only.py` covers snapshot ordering, exact closures, capability
schemas, byte-worker rejection cases, deterministic output hashes, the exact
2.5-bpw fixture ledger, and preservation of the pinned v2 core. It is frozen
but not executed here because this environment has no Python runtime.

No network or payload launch is authorized by this package.
