# Independent UWFA-SC v2 source audit — 2026-09-02

This is a source-only, independent audit of
`research/unifilar_wfa_entropy_census_stage0_v2` at producer commit
`95602d85c234a95bf2f240d7c093ee9f26e25e42` and exact
`SOURCE_MANIFEST.json` SHA-256
`223a96585444a0b3e4344c470e243dbd4b84662fddfda881185e879a4caee693`.

The audit did **not** open, stat, hash, enumerate, or otherwise access Qwen
weights, the current finite artifact, extracted SC streams, or Gaussian
controls. It grants no payload authority.

## Verdict

`BLOCK_UNIVERSAL_LAUNCH_AND_PAYLOAD`

The causal UWFA mechanism is credible and the small all-150 CPU/CuPy equality
fixture passes on the supplied RTX 5090. The sealed source nevertheless cannot
be launched as the requested universal SwiGLU-MoE codec because its physical
ownership ABI cannot represent a 128-expert deployment, its parser accepts
unbounded/excess expert counts before expensive shifts and loops, and the
mandatory representative runtime/memory preflight is absent.

## Exact hard counterexamples

The reproducer in `reproduce_blockers.py` authenticates every producer source
member before compiling the exact buffered bytes.

1. A source-free six-expert fixture builds and parses.
2. Re-sealing only the authenticated container header with `experts=4097`
   is accepted by `parse_container`, although `protocol.MAX_EXPERTS=4096`.
   The parser reaches `_owners`, which forms `1 << experts` and iterates
   `range(experts)` without first enforcing any frozen bound.
3. Building a 128-expert region owned by expert 127 (`owner_mask=1<<127`)
   fails with:

   ```text
   error: 'I' format requires 0 <= number <= 4294967295
   ```

   The region, frame, and directory owner masks are little-endian u32. The
   inherited STRATA block record is even narrower: u16. There is no owner-map
   indirection or grouping grammar that could legally recover expert ordinals
   32..127 (or 16..127 in inherited metadata).
4. A container with `experts=128` and one low owner mask is accepted; 127
   declared experts own no stream. Thus using a low-numbered group alias is not
   a specified escape: it silently loses expert identity and violates the
   nonempty-expert/owner-attribution contract.

The actual current six-expert artifact was not accessed. The source-free
six-expert path establishes only format reachability, not a Qwen result.

## Mechanisms that passed

- The pinned producer manifest and every member byte count/hash match.
- No undeclared producer member was present.
- Native source verifier passed.
- All 23 source-only hostile tests passed in 30.909 seconds.
- Direct `stage0_census.py` execution is inert and exits 2.
- The adapter exposes `set_level(level)` and `decode(original_freq1)`, resets
  before emission, derives the public prior bin inside SC, reads a frequency
  from the serialized model, and transitions only after the decoded bit.
- The literal container parser validates model serialization, frame/payload
  hashes, canonical padding, reconstruction binding, and re-encoding paths.
- The source/control selection code repeats the 150-cell search per control,
  and the final positive predicate is an explicit conjunction.

These passes do not override a hard physical-format counterexample.

## RunPod CPU/CuPy result and why the preflight remains unmet

The producer's built-in `gpu_preflight_all_150` passed on the supplied RunPod:

- CuPy 14.2.0, CUDA runtime 12.9, driver 13.0;
- NVIDIA GeForce RTX 5090, compute capability 12.0;
- 150/150 cells, repeated GPU runs equal to the CPU reference;
- four streams of lengths 4097, 2053, 1031, and 521 (7,702 symbols total);
- elapsed 13.950451377080753 seconds in the sealed receipt run;
- 750 kernels, 2,310,600 count updates and 3,465,900 length updates.

This is an arithmetic-equivalence microfixture, not the mandatory feasibility
preflight. It has four streams, no complete synthetic outer fold, no nested
150-cell selection/refit workload, and no representative production-length
panel.

The telemetry also undercounts H2D traffic. The reported 30,872 bytes are
exactly the initial four packed streams (`4*7702`) plus four 16-byte descriptor
records. `exact_lengths` calls `cp.asarray(frequencies, dtype=cp.uint16)` for
every model transfer without incrementing `h2d_bytes`. In this microfixture the
450 length-kernel calls transfer another 7,257,600 model-table bytes, over 235
times the reported total. No peak host-RAM or peak-VRAM sampler exists; the
receipt contains only instantaneous/free and memory-pool values.

Consequently these mandatory checklist gates remain unmet even though the
small equality check itself passes.

## Independent bridge comparison

The frozen reference bridge manifest is
`51f158c7f82fad81bd2b15d30e6581a2847e0e436d98f085055b8d818bf43f31`;
its independent audit manifest is
`cb23a8638f49fd69cdb6f518b089ec08c5dbf87a9749d695acbf37551cf124c7`.
That audit also classifies mechanism tests as PASS and launch as BLOCK.

The two implementations agree on the semantic adapter ABI:

- level is set before a selected SC decision;
- `original_freq1` is regenerated causally rather than transmitted per symbol;
- state resets before the first emission in a reset interval;
- a serialized model selects the new arithmetic frequency;
- state transitions after the bit; and
- decoded decisions must canonically re-encode.

They are not byte-compatible independent decoders:

- producer container magic is `UWFCV2\0\0`; bridge magic is `UWFASC2\0`;
- producer directory/frame sizes are 160/128 bytes; bridge sizes are 256/64;
- producer model context uses `(level, prior_bin, position mod 4, state)`;
  bridge serializes `(level, prior_bin, full position in reset, state)`;
- producer uses one of five procedural transitions; bridge freezes a different
  topology; and
- producer owner masks are u32 (u16 in inherited records), while the bridge
  directory uses u64 and then deliberately enforces a six-expert topology.

Therefore the bridge corroborates the causal mechanism only. It cannot parse
or independently validate a producer container, and it supplies no 128-expert
escape.

## Required repair before another launch audit

At minimum, a new sealed source version must:

1. replace fixed masks with a bounded, canonical owner-set representation that
   covers the declared expert universe (including 128 experts), or define and
   fully charge a canonical indirection table;
2. enforce exact experts/streams/weights/offset bounds before every shift,
   iteration, allocation, or GPU conversion;
3. reject every declared expert without the required nonempty owned streams;
4. generalize the fixed six-expert/fifteen-block/768x2048 adapter or narrow all
   universality claims to this diagnostic panel;
5. provide a separately pinned external dispatcher and exact independent
   decoder for the producer's real byte ABI; and
6. run a representative complete synthetic outer-fold benchmark with complete
   H2D, peak host-RAM, peak VRAM, kernel, and wall-time telemetry.

No Qwen or control payload may be opened under this source manifest.
