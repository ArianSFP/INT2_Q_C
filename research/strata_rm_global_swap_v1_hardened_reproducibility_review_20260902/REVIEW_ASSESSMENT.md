# Independent software-quality and reproducibility assessment

## Outcome

The frozen v1 producer materially repairs the v0 source-closure, arbitrary
hook-injection, schema-only packet, and in-process CuPy-facade defects.  Its
serialization and byte-recomputation design is substantially reproducible.

It is still correctly labelled source-only and unexecuted.  The present
physical contract is not, by itself, sufficient authority for a Qwen,
matched-Gaussian, universal-SwiGLU, or measured-read result.  That conclusion
does not invalidate the RM-order mechanism; it constrains what a future result
may claim and what an independent launcher/auditor must add.

## Properties verified by frozen source inspection

| Area | Review finding |
|---|---|
| Source closure | The manifest is externally pinned, lists exactly 14 regular members, and commits canonical member metadata to source root `980a5f1d...d4f14`. `authority.authenticate_v1_package` rejects a linked package root, unmanifested entries, links, and member mutation. |
| Dependencies | The v0 producer, v0 independent-audit source, current encoder, BEC wrapper, and historical independent decoder are bound by fixed SHA-256 values and orientation checks. This authenticates source, not execution receipts. |
| Serialization | `canonical_json` uses sorted ASCII keys, compact separators, and `allow_nan=False`; `strict_json` rejects duplicate keys and non-finite constants. Experiment commitments additionally require exact canonical bytes plus one newline. |
| Worker isolation | `run_isolated_worker` copies every authenticated v1 member into a fresh temporary snapshot, reauthenticates it, invokes `sys.executable -I -B`, drops Python import environment controls, closes stdin, and reads a literal output receipt. |
| Hook binding | The current integration worker accepts no module or hook object. It loads exact pinned files, checks `bg.base is base`, checks the live `bg.bec_flags` reference, installs the local closure, and checks final hook object identity. |
| Packet consistency | The synthetic decoder consumes a length-delimited packet with magic, canonical header, exact end offset, and CRC; writes literal FP64 reconstructions; regenerates canonical bytes; and the parent compares them with the original packet as bytes. |
| Numerical recomputation | BF16 source bytes are expanded deterministically, FP64 reconstruction bytes are decoded explicitly, non-finite values fail, per-matrix SSE/energy are accumulated with `math.fsum`, and rate, relative MSE, and `F` are derived rather than accepted from the caller. |
| Accelerator provenance | The worker requires a fresh isolated interpreter, rejects preloaded NumPy/CuPy, rejects controlled-root import origins, checks live device/runtime/driver values, synchronizes a GPU arithmetic probe, and compares complete N20/N21 orders. |
| Scope | `EXECUTION_STATUS.json` says all Python, hook, CuPy, and payload runs are unexecuted; `payloads_opened=0`; `qwen_result=null`; and every production claim remains held. |

These are source findings until the frozen review test receipt and, separately,
the real-CuPy receipt execute successfully.

## Correctness and authority gaps

### 1. Standalone verifier resolves away a package-root link

`verify_source.py` calls `resolve(strict=True)` before testing
`package.is_symlink()`.  A supplied root link therefore becomes its real
target and passes the standalone verifier.  The stronger
`authority.authenticate_v1_package` checks `lstat()` before resolution and
does reject the link.  Static-only callers should use the stronger primitive,
or the standalone verifier should preserve and check the original path.

### 2. Current external modules are authenticated but not snapshotted

The v1 worker package is snapshotted correctly.  The pinned current base and
BEC files are hash-checked and then imported from their original external
paths.  A concurrent mutation between authentication and import is therefore
a hash-to-use race.  The worker reports post-import file hashes, but the
parent's generic receipt check does not compare those fields to the pins.
Copy the three external sources into the authenticated worker snapshot, or
revalidate exact bytes immediately after execution and validate the
worker-specific receipt schema.

### 3. The physical decoder is not executed from an immutable snapshot

`validate_physical_bundle` hashes the decoder worker and later executes that
external path directly.  Unlike `run_isolated_worker`, it does not snapshot
the authenticated worker.  A production authority launcher should execute a
private immutable copy or use a file-descriptor/OS mechanism that closes the
hash-to-exec race.

### 4. Decoder-audit independence is not established out of band

Production mode requires an exact audit directory and binds it to the worker
hash, which is useful.  But the audit manifest hash and audit source root are
both supplied inside the same experiment commitment as the worker.  No
separate auditor pin, audit receipt, minimum audit protocol, or successful
execution status is required.  The `independent_from_encoder` field remains a
boolean assertion.  A true production gate needs an out-of-band approved
decoder-audit root and a frozen successful audit receipt.

### 5. Decoder file access and read accounting are trusted reports

The decoder is not filesystem-sandboxed.  `source_payloads_opened=False` and
the packet read operations are fields emitted by the decoder, and the parent
validates their schema/arithmetic.  It does not independently instrument file
opens or byte reads.  A separately audited simple worker may make that trust
reasonable, but the current validator alone does not measure it.  Production
read authority needs OS tracing/sandboxing or a minimal audited reader whose
only supplied descriptor is the expert packet.

### 6. Model, Qwen, control, and universality identity are declarative

Source rows are strongly byte-pinned, but `kind`, `architecture_family`,
`pipeline_id`, `qwen_specific_tables`, and `model_family_agnostic` are labels
in the commitment.  There is no authenticated checkpoint/model manifest,
tensor-name provenance, or independently frozen Gaussian generator/seed and
moment record.  Geometry proves a SwiGLU-compatible triplet, not Qwen identity
or cross-family portability.

### 7. Cross-family evidence does not have to pass the target

Production target rate, `F`, and maximum read amplification are pooled only
over `qwen` cases.  The second architecture family must be present in
`model_cases`, but its rate, `F`, and read amplification are not acceptance
criteria.  A universal-codec claim should require the sealed criteria for
every held-out architecture family, with per-family results.

### 8. Matched controls are present but do not affect acceptance

The validator requires the paired control to have equal geometry, weight
count, and pipeline ID.  It does not authenticate Gaussian generation or use
Qwen-minus-control advantage in any acceptance calculation.  Controls are
therefore bookkeeping evidence, not yet a scientific matched-control test.

### 9. Hook installation and GPU parity are not payload integration

The current-hook worker authenticates installation but never invokes the hook
at N20/N21.  The v0 independent audit source is authenticated, but its frozen
status was unexecuted.  Likewise, the real-CuPy worker's CPU and GPU functions
come from the same `rm_order.py`; the optional review adds an independently
constructed order hash, but a real receipt is still required.  Neither result
would authorize payload access by itself.

## Required production closure

Before any Qwen or universal claim, a future gate should add:

1. an out-of-band pinned, successfully executed independent decoder audit;
2. immutable snapshots of external integration modules and the physical
   decoder worker;
3. authenticated checkpoint/tensor provenance and independently generated
   matched controls;
4. instrumented packet-only decoder I/O and routed-read measurement;
5. target checks per architecture family, not mere family presence;
6. successful frozen source, hook, independent-order, and real-CuPy receipts;
7. one literal packet and original-source reconstruction audit.

## Disposition

```text
SOURCE_DESIGN_SUBSTANTIALLY_HARDENED__REPRODUCIBILITY_GAPS_RECORDED__PYTHON_AND_CUPY_UNEXECUTED__HOLD_PAYLOAD_AND_PHYSICAL_RESULT
```

No row-overlap result, source fixture, hook-install receipt, or GPU-order pass
may be presented as Qwen MSE, physical bpw, `F`, matched-control advantage,
universal portability, or routed-read evidence.
