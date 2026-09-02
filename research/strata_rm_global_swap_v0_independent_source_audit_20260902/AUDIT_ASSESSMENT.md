# Independent audit: STRATA global RM-order swap v0

## Scope and pinned object

This audit is source-only and opens no Qwen, Gaussian-control, coarse, or
model payload.  It independently pins producer source root
`4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c`
and source-manifest SHA-256
`939b57518a4afe11c56c59a20f109e423c8eab5815c947cb3e5a91d559704b3c`.

The stated mechanism is narrow: retain the six actual selected counts `K`
from the current BEC hook, replace only the internal-phase ordering by
descending generator-row weight with an ascending-phase tie break, keep the
current random frozen-value coset, and permit only global lengths `2**20` and
`2**21` through `install()`.

## Mechanism assessment

The independent tests derive the Arikan generator by Kronecker products and
verify that internal phase `i`, after the pinned bit reversal, generates a row
of weight `2**popcount(i)`.  They compare the complete small-N order to an
independent `(-popcount(i), i)` sort, exercise both production lengths, and
require exact per-level equality to the counts returned by the called
reference hook.  They also verify the current-random RNG convention and the
hard separation of the held zero coset.

This establishes the source mechanism only if the executable test receipt is
successful.  It cannot establish a Qwen rate-distortion result.

## Authority findings

1. **The producer source verifier ignores unmanifested directories.** It
   enumerates only top-level files.  An importable `cupy/` or `numpy/`
   directory can therefore be added while the producer verifier still passes.
   `independent_auth.py` closes this gap by requiring the exact complete
   top-level entry set and rejecting symlinks.

2. **The hook installer is not integration authority.** `install()` accepts
   any object carrying a `reliability_freeze_flags` attribute and any callable
   reference hook.  It authenticates neither object identity nor the final
   monkeypatch after later imports.  This is consistent with the producer's
   stated integration HOLD, but a future launcher must bind the current global
   encoder, reference hook, import order, final hook identity, and independent
   decoder before payload access.

3. **The result contract is a schema checker, not physical evidence.** A
   fabricated receipt with no packet bytes passes.  Decoder and packet hashes
   are checked only for non-emptiness or 64-character equality; booleans assert
   independent decode, causal regeneration, exact consumption, and canonical
   replay without performing them.  The contract contains no authenticated
   source identity, decoded reconstruction, SSE/energy, relative MSE, `F`,
   matched controls, expert locality, routed reads, or universal SwiGLU-MoE
   evidence.  Literal byte arithmetic and the `[2.15,2.5]` interval are
   correctly recomputed, but they are arithmetic over self-declared counts.

4. **The producer CuPy smoke is spoofable.** It accepts an injected NumPy
   facade as `cupy` and emits its PASS status.  The separate
   `run_real_cupy_audit.py` must be launched in a fresh isolated interpreter;
   it rejects experiment-controlled import origins, checks a live CUDA
   runtime/driver/device, and compares the full `2**20` and `2**21` GPU orders
   to an independently constructed NumPy order.  Even that is mechanism-only
   evidence.

5. **No universal or read claim is implemented here.** The source package
   contains no expert packet, decoder, tensor-role traversal, cache layout, or
   cold-page measurement.  Those claims correctly remain held.

## Disposition

`PASS_STATED_SOURCE_MECHANISM__HOLD_PAYLOAD_INTEGRATION_AND_PHYSICAL_RESULT`

That disposition is available only after the frozen hostile suite executes
successfully.  Before execution, the correct status is
`FROZEN_INDEPENDENT_AUDIT_UNEXECUTED__HOLD_RUNPOD_AND_PAYLOAD`.

No overlap percentage, selected count, ideal entropy, fixture receipt, or
source-only PASS may be presented as MSE, bpw, `F`, Qwen, Gaussian-control,
universal-codec, or routed-read evidence.
