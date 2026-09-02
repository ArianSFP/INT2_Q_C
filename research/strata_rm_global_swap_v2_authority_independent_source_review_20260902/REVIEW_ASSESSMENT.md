# Review assessment

## Scope and pins

- Producer manifest SHA-256: `1f1caf2884a8b0b8713f213a16a0a32194238b64969e9d9cf3aaa339ddb776be`
- Producer source root: `e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9`
- Frozen v1 producer root: `980a5f1d272ca5ffc7b4d35e7c234a86994d135fcacaf0d47a8b3e00fc3d4f14`
- Frozen v1 review root: `1dfa55969b87543adbee785d72933f9ccb6f754eaade9e4e340a022c96c1afa8`

This review examined every producer member. It made no payload or network
access and recomputed the flat closure using an independent PowerShell
implementation.

## The nine v1 findings

1. **Root link resolution: repaired in source.** `real_directory` and the
   standalone verifier use `lstat` and reject the caller's root before
   `resolve`. Member traversal rejects symlink components.
2. **External hash-to-import race: substantially repaired.** The three pinned
   external files are copied into a fresh snapshot and the current worker
   imports the base and BEC modules from that snapshot. The parent rehashes the
   snapshot after the worker. This is a sound benign trusted-runner mechanism;
   chmod alone is not an adversarial immutability primitive when execution is
   as root.
3. **Decoder hash-to-exec race: substantially repaired.** Exact decoder and
   launcher bytes are copied to a fresh directory, executed from those paths,
   and rehashed after execution.
4. **Decoder audit self-declaration: repaired at the capability boundary.** A
   separately pinned, exact regular-file audit closure and separately pinned
   executed PASS receipt are mandatory and bind both worker and launcher
   hashes. This is still a trust boundary, not cryptographic proof that the
   named human/process was independent.
5. **Decoder-reported read counts: only partially repaired.** The launcher owns
   the reported intervals and ignores decoder-reported counts, which is an
   improvement. However, `PacketReader.__getattr__` exposes the underlying
   buffered handle (`raw`, `peek`, `readinto1`, `fileno`, and other methods),
   while `os.read`, `mmap`, `ctypes`, native library I/O, and imported-extension
   I/O are not instrumented. A decoder can perform one instrumented full read
   and additional uncounted reads. These are logical Python stream intervals,
   not physical page or device reads.
6. **Model/control/family declarations: not fully repaired.** The experiment
   commitment can no longer declare these labels, but the separately pinned
   scientific capability still self-declares `owner`, an audit receipt hash,
   an auditor source-root hash, execution status, checkpoint identity, tensor
   identity, family, matched-control generator, moments, and selection facts.
   Unlike `authenticate_decoder_audit_capability`, the authority does not open
   or authenticate that audit closure or receipt. The move is useful separation
   of roles, but it is not authenticated independent scientific provenance.
7. **Cross-family target enforcement: numerically repaired, provenance still
   incomplete.** `evaluate_family_acceptance` applies rate, F, read, and source-
   specific gates independently to every declared family. Yet two different
   family strings may reuse the same checkpoint, tensor manifest, source rows,
   and bytes; distinct architecture families are not mechanically established.
8. **Controls affect acceptance: repaired mathematically, fidelity remains a
   scientific-capability hold.** The strongest complete declared control pool
   is subtracted in `s=-0.5*log2(F)`, and at least 0.03 bpw is required. Control
   rate and read gates are applied. Moment matching and complete replay are
   assertions in the unauthenticated scientific capability described in item 6.
9. **Hook and RM parity: repaired in source, execution pending.** The current
   snapshot worker installs and invokes the global hook for `2**20` and
   `2**21`. The parity worker compares the frozen v1 implementation with an
   independent CPU Gosper enumeration and a CuPy byte-LUT implementation. The
   current frozen execution status correctly says these workers are unexecuted.

## Additional blocking issue: routed expert reads are not measured

The user's inference constraint is routed cold-read amplification per expert.
The current physical API instead decodes every source listed in a capability
case, requires the decoder to cover every byte of the case packet, and divides
logical bytes read by the whole packet length. `_validate_source_rows` permits
many `(layer, expert)` triplets in one case. Consequently, a case containing a
whole layer can read the whole-layer packet once and report exactly `1.0x`,
even though serving one routed expert would require reading far more than that
expert's compressed bytes. There is no selected-expert request, expert-local
packet interval map, page rounding, shared-stream accounting, or maximum over
independently routed experts.

This is not a cosmetic audit detail. The claimed `<2x` condition is not the
requested `<2x` per-expert cold-read condition. A future authority must require
either one expert triplet per packet/case or an authenticated expert-local
index, invoke the decoder once per routed expert, and compute physical page
reads relative to that expert's literal compressed allocation (including all
shared/common pages charged under the declared cache policy).

## Correctness observations

- The CPU RM order is descending popcount and ascending phase, implemented by
  fixed-weight Gosper enumeration. The CuPy implementation independently uses
  a byte popcount table and unique integer sort keys. Their specified orders
  agree mathematically for both production widths.
- Current-hook replacement preserves the selected count at all six levels.
  The current integration test is a hook-level source test with fixed synthetic
  capacities, not a full encoder/arithmetic payload execution.
- Rate, BF16-source/FP64-reconstruction SSE, pooled relative MSE, F, and saving
  are derived from literal bytes. Caller-supplied metrics are not accepted.
- Exact packet canonical replay is required.
- The source package honestly carries `HOLD_PAYLOAD_AND_RD`; no Qwen or finite
  codec result follows from this source review.

## Required repairs before physical authority

1. Authenticate the independent scientific-audit closure and executed receipt
   exactly as strongly as the decoder audit, with those pins supplied out of
   band rather than embedded only inside the scientific capability.
2. Bind every checkpoint/tensor/source identity to that audited closure and
   reject cross-family source/checkpoint aliasing unless explicitly justified
   as the same architecture family.
3. Replace the whole-case read ratio with a per-routed-expert decode protocol,
   expert-local interval ownership, physical page rounding, and a maximum over
   experts/families.
4. Instrument reads below the Python buffered stream or constrain and audit the
   exact decoder API so unsupported handle/native I/O cannot bypass accounting.
5. Execute the frozen Python/CuPy source gates and record external receipts
   before promoting the source mechanisms.

## Disposition

The v2 package is a material improvement and closes the RM-order/source-snapshot
mechanics well enough to justify executing its no-payload gates. It is not yet
a sound physical-result authority for universal SwiGLU-MoE or routed expert
bandwidth. Payload and RD execution remain held.

```text
PASS_STATIC_CLOSURE_AND_SUBSTANTIAL_V1_REPAIRS__BLOCK_PHYSICAL_AUTHORITY_ON_SCIENTIFIC_PROVENANCE_AND_ROUTED_EXPERT_IO__HOLD_PAYLOAD_AND_RD
```

