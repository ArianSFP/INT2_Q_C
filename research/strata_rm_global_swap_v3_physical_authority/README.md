# Global STRATA RM swap v3 physical authority

Date: 2026-09-02

Status: **frozen source-only; runtime, payload, and all RD claims held**.

This package repairs the blocking findings in the independent review of v2.
It pins v2 source root
`e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9`
and independent-review root
`d642889efcf8c54173eb7659602181cb9e71e122ce11ff05da6b24e45c47a113`.
It changes no quantizer, selected set, reconstruction, or RM ordering.

## Scientific provenance is now an authenticated audit package

A standalone JSON object cannot establish Qwen identity, checkpoint identity,
matched controls, selection discipline, or architecture portability. V3
requires an exact independent scientific-audit closure containing:

- frozen audit source;
- a canonical scientific capability;
- a separately hash-pinned executed PASS receipt;
- exact checkpoint, tensor, source, generator, seed, moment, family, and
  selection-replay attestations.

The audit manifest, audit-source root, receipt SHA-256, and capability SHA-256
are all supplied out of band to the physical entry point. Cross-family reuse
of checkpoint manifests, tensor manifests, checkpoint identities,
architecture schemas, source paths, or source byte hashes fails closed.

## The physical unit is one routed expert

Every route contains exactly three tensors belonging to one `(layer, expert)`:
Gate, Up, and transposed-compatible Down. Every route owns one distinct packet
file; packet paths and byte hashes cannot be shared. No common or shared stream
is admitted by this version.

The authority independently launches one decode per route. It rounds the
expert packet to 4 KiB pages and records:

- literal packet bytes supplied;
- exact page indices and literal bytes on each page;
- zero padding in the final page;
- physical page bytes supplied;
- cold-read amplification = physical page bytes / literal expert packet bytes.

The maximum is taken across independently routed experts and then enforced for
every claimed architecture family. `R in [2.15,2.5]`, `F<=0.8`, `<2x` cold
reads, strongest-control subtraction, and absolute Qwen `F<=0.8` remain
mandatory.

## Decoder isolation

V2's Python handle wrapper could be bypassed through raw handles or native I/O.
V3 therefore permits only a separately audited WebAssembly decoder with zero
imports. It is instantiated without WASI and without a linker. The decoder has
no path, file descriptor, callback, environment, clock, random source, socket,
subprocess, ctypes, dynamic-library, or native-read capability.

The trusted sandbox host places a page-padded copy of the already opened expert
packet in private linear memory. The decoder sees only that buffer and scalar
geometry. Mutation of the input region fails. Exact canonical packet replay is
required. The sandbox source and decoder module are independently pinned and
copied to fresh read-only snapshots before execution.

## Source-only gate

```bash
python -I -B run_source_gate.py \
  --package /workspace/strata_rm_global_swap_v3_physical_authority \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --v2-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v2_authority \
  --review-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v2_authority_independent_source_review_20260902 \
  --output /tmp/strata_rm_global_swap_v3_source_gate.json
```

This command accepts no checkpoint, tensor, packet, decoder, scientific audit,
or model path. The physical entry point remains unusable until independently
frozen successful scientific and decoder audit packages exist.

```text
FROZEN_V3_PHYSICAL_AUTHORITY_SOURCE_ONLY__HOLD_WASMTIME_RUNTIME_PAYLOAD_AND_RD
```
