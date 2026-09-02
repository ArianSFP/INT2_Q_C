# Independent source audit: LOGIC-Q v2 bound adapter

Date: 2026-09-02

Disposition:

```text
MECHANISM_VALID__HOLD_PRODUCTION_PROVENANCE_BACKEND_AND_STRATA
```

The externally pinned source closure is:

```text
SOURCE_MANIFEST.json SHA-256
e97041b2debdd1a85ce32305f43aae1f76cf4ca937b52e275bdd246ae1b1b980

source root SHA-256
080de7a63e596ae34f9da90941d7fd9d07b70dfb2afad97103aa5ab5943d3776
```

This is an independent audit directory.  The frozen v2 package was not
modified.  No model, Qwen tensor, current STRATA packet, coarse packet, or
matched-control payload was opened, statted, hashed, enumerated, or inferred.

## Outcome first

The low-level abstract four-level mechanism is sound on literal bytes:

- exact source closure and pinned v1/v0 dependencies verify;
- an actual packet is canonical-decoded and canonical-re-encoded;
- source counts and physical bits can be derived from its literal headers;
- positive finite BF16 scales, zero internal/final padding, role order, equal
  Gate/Up/Down-transposed shapes, and page closure are enforced;
- the separate scorer authenticates literal BF16/FP32/FP64 source blobs and
  recomputes unweighted SSE and source energy in FP64;
- the source-only fixture is one page-contiguous expert object and is therefore
  *addressable* with a one-fetch layout.

V2 nevertheless cannot authorize a production result.  Its orchestration
boundary still accepts attacker-created evidence.

## Critical findings

### 1. Scored rows are assertions, not authenticated measurements

`authorize_test` correctly recomputes the winner and aggregates from the rows
it receives.  It does not establish where those rows came from.  A row is
authenticated only by a public SHA-256 self-seal and the literal string
`logic-q-v2-independent-source-scorer-v1`.

The hostile audit constructs every row without invoking the scorer, assigns
arbitrary canonical FP64 SSE/energy strings, reseals them, supplies the new
receipt hash as the external pin, and is authorized.  This is not the narrow
post-pin mutation attack fixed from v1; it is creation of a completely false
pre-pin experiment.  An external hash is a commitment once independently
fixed.  It is not provenance, a signature, or evidence that a trusted scorer
observed the bytes.

### 2. Packet receipts omit the packet

Selection receives a compact packet-geometry receipt, not packet bytes.  It
reparses copied header hex, but receives neither scales nor label/model payload.
Arbitrary expert and component SHA-256 strings validate because there are no
bytes from which to recompute them.  Consequently:

- scale and payload legality proven by the real decoder do not transfer to the
  selection receipt;
- physical/model bytes can be asserted rather than independently observed;
- one invented packet can be aliased to every layer, slot, and config;
- a packet receipt's declared `1.0x` is a topology assertion, not a measured
  runtime storage-read trace.

The actual canonical packet has no embedded CRC.  A bit flip that changes a
literal label can remain a perfectly canonical packet.  Integrity therefore
depends on an authenticated external content hash.  V2 has a hash value, but
the authorizer has no packet bytes and no trusted party binding that value to
the selected measurement.

### 3. Config identity is not bound to packet bytes

The independent scorer checks that `config_id` belongs to the frozen grid, but
the abstract component/expert packet does not serialize or derive that config.
The same literal packet scores under multiple config IDs.  Selection can thus
be forged by reusing identical bytes for the whole config bank and changing
only asserted metrics.

The launch boundary has a related gap.  `make_launch_context` accepts any
well-formed selection-receipt hash and any config string; it does not parse the
referenced receipt or prove that the config was selected.  `encode_expert_bound`
receives neither the selection receipt nor a capability returned by
`authorize_test`.  Its check only compares the caller's config object to the
caller's launch context.

### 4. The CuPy probe is an environment check, not an attestation

Rejecting a name-only object and a bare module shell is useful against mistakes,
but it does not authenticate a hostile Python process.  The hostile audit puts
a complete `types.ModuleType("cupy")` facade in `sys.modules`, gives it a valid
`ModuleSpec`, a regular `__file__`, NumPy implementations of the arithmetic
probe, and fake CUDA runtime/device objects.  Both receipt collection and fresh
validation accept the CPU facade.

This is inherent to in-process self-attestation: code under the caller's
control cannot prove to itself that its imports or device API have not been
replaced.  V2 should describe the current check as accidental-backend
protection, not adversarial attestation.

### 5. Holdout identifiers are whole, but content aliasing is open

The panel rebuild correctly holds out whole test layers and whole validation
expert slots; component ordinals and role triplets are canonical.  However,
identical source hashes are allowed across train, validation, and test.  The
hostile panel uses the same nonexistent digest for every component and passes.
Thus identifier-level partitioning is implemented, but duplicate content and
owner aliases are not closed.

### 6. Controls and final test have no bound result gate

V2 has no executable path which consumes literal matched-control bytes and
packets, independently scores them, aggregates whole-owner uncertainty, and
authorizes the absolute final `F` result.  The inherited v1 control generator is
useful source-only machinery; README promises do not bind future output.

## Read-bandwidth interpretation

For a real canonical fixture, all three components live in one page-contiguous
expert object and all bytes are charged.  This proves a `1.0x` *minimum routed
storage fetch layout*.  It does not prove observed `1.0x` storage, HBM, or DRAM
traffic.  The receipt sets `read_passes=1` and
`routed_storage_read_bytes=packet_bytes` by construction; no runtime counter or
I/O trace is consumed.  A production claim should distinguish:

```text
layout-addressable read amplification = 1.0x
measured runtime read amplification    = not measured
```

Decoder scratch and materialization traffic should be reported separately
from compressed-storage traffic.

## STRATA semantic critique

Three different objects must remain separate:

1. **Frozen v2 abstract four-level packet.**  One of four profile levels is
   selected at each weight.  It is a useful standalone mechanism fixture only.
2. **Proposed direct STRATA-RM6 block.**  The plan uses six RM/sub-RM Boolean
   planes on a new `N=4096=2^12` block and joins them into one index `0..63`.
   Six `RM(5,12)` dimensions total 9,516 bits, or `2.3232421875 bpw`, before
   every header, scale, selector, exception, CRC, alignment, and page byte.
   This can become a new direct packet only after a literal decoder and ledger
   exist.
3. **Current STRATA packet.**  The deployed geometry contains `N=2^21` and
   `N=2^20` polar blocks.  It performs six complete level-major SC passes;
   completed polar output planes form the `0..63` index.  The causal decisions
   are Q0.16 arithmetic-coded.  Selected SC positions, RM dimensions, and
   physical emitted bits are therefore different quantities.

The plan correctly says that the 4096-site construction is a new
`STRATA-RM6` family unless it also inverts the polar transform, enforces the
current frozen positions, replays current causal arithmetic, and reproduces
the existing packet.  A coordinate-local four-level result has no Qwen or
STRATA implication.  Likewise, swapping a reliability order in the current
large polar code yields an RM-ordered truncated polar code unless its active
rows exactly match a complete RM dimension; it is not automatically the direct
4096 RM6 codec.

The direct RM6 plan is mathematically coherent, but it remains a design.  Its
724-bit nominal remainder at 2.5 bpw is not an operational margin until all
literal per-block and expert-level bytes are emitted.  Its exact 64-way costs
must be jointly rescored after assembling all six planes, through the original
source-domain inverse path.  Plane-wise gains cannot be added.

## Narrow v3 repair

Do not enlarge the algebraic search yet.  Repair the evidence boundary:

1. Run selection in an auditor-owned process which receives the externally
   pinned panel and literal raw source and packet blobs.
2. Parse, canonical-decode, canonical-re-encode, and independently reconstruct
   every packet in that process.  Derive all packet/component hashes, source
   hashes, counts, rates, scales, payload lengths, SSE, energy, and rows
   internally.  Do not accept any of them from an encoder receipt.
3. Pin the panel, frozen grid, scorer closure, and selection protocol before any
   train/validation rows are admitted.  After scoring, externally pin the
   complete content-carrying selection artifact before test bytes are opened.
4. Bind config identity to a canonical packet field or to an auditor-owned
   encode invocation whose packet hash is returned directly; forbid the same
   packet/config claim from being relabelled.
5. Serialize/authenticate scale and payload bytes, not only header hex.  Add a
   physical CRC/checksum if packets must detect corruption without the external
   manifest.
6. Import CuPy in a fresh restricted process, before untrusted modules execute;
   query module/distribution closure and CUDA device there, execute and
   synchronize a real kernel, and return the packet directly from that process.
   Treat this as trusted-runner evidence, not cryptographic self-attestation.
7. Make the launched config a direct output/capability of authorization, and
   have the launch process reparse the pinned selection artifact.
8. Reject duplicate source hashes across disjoint owners unless an explicit,
   audited alias map proves why they are the same object.
9. Build separate literal control and whole-test result gates.  They must score
   bytes through the same trusted path and cannot create an absolute pass.
10. Keep Qwen closed until a separately audited STRATA-RM6 semantic adapter
    exists.  The four-level pilot may run only under that exact label.

## Source-free RunPod observation

A generic real-device probe was permitted without uploading repository source:

```text
CuPy                         14.2.0
device                       NVIDIA GeForce RTX 5090
CUDA runtime                 12090
CUDA driver                  13000
cupy/__init__.py SHA-256     8c4724758587dea5f1c1d7c217c74a9fa0e4ed7f9d76a2b86fa001117cf3c718
probe observed               126727726
probe expected               126727726
stream synchronized          true
```

The complete frozen-v2 hostile script subsequently ran locally under Python
3.12/NumPy and passed all 27 mechanism/attack checks; its literal output is
`HOSTILE_AUDIT_RESULT.json`.  The managed escalation reviewer denied export of
the frozen repository source packages to the external RunPod, so it was not run
there.  This audit must not represent the generic device observation as a v2
launch receipt.

## Claim boundary

This audit validates useful source-only four-level packet and scorer mechanics.
It does **not** validate a Qwen result, a current STRATA result, a matched-control
result, `F <= 0.8`, runtime `1.0x` measured bandwidth, or a universal SwiGLU-MoE
codec.
