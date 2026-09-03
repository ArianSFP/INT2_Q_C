# Global STRATA RM swap v4 Wasmtime authority

Date: 2026-09-03

Status: **frozen source-only; Wasmtime execution, payload, and all RD claims held**.

This final narrow repair pins v3 source root
`83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad`
and its independent-review root
`3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0`.
It changes no quantizer, reconstruction, selected family, source panel, or
acceptance threshold.

## Complete Wasmtime provenance

Production execution now requires a separately pinned, independently executed
runtime-audit package. Its external authority tuple is:

- audit manifest SHA-256;
- audit-source root SHA-256;
- successful receipt SHA-256;
- runtime-capability SHA-256;
- complete runtime-tree root SHA-256.

The exact Python distribution version, runtime version, Python ABI, wheel
platform, compilation target, every Python module, all resources, distribution
metadata, and every native library are copied and hashed. Empty surplus
directories and unmanifested files fail. The sandbox authenticates that tree
before importing `wasmtime`, requires the imported module to originate in the
snapshot, and verifies the mapped native libraries against the pinned files.

## Bounded engine before guest execution

The engine enables fuel consumption before compilation. Each decoder and
independent encoder receives a new Store whose limits are installed before
instantiation:

- linear memory: at most 1 GiB;
- table elements: at most 10,000;
- instances: at most 2;
- tables: at most 2;
- memories: at most 2;
- fuel: `min(10^12, 10^8 + 50,000 * literal_packet_bytes)`.

The independently executed runtime receipt must additionally prove that both
the limiter and fuel-exhaustion probes work for the exact frozen distribution.
The parent process retains a wall-clock timeout.

## Architecturally immutable packet input

The trusted host opens one routed-expert packet as an immutable Python `bytes`
object before importing Wasmtime. The decoder receives no packet pointer,
path, file descriptor, WASI interface, native-read primitive, or mutable host
buffer. Its sole import is:

```text
authority.read_packet(offset, destination, length) -> status
```

The callback range-checks every request, rejects overlap, copies only that
slice into guest-owned memory, and records it. Success requires a disjoint
partition covering every literal packet byte exactly once. The receipt records
all `(offset,length)` operations, all touched 4 KiB pages, literal bytes, and
page-rounded physical bytes. This is architectural isolation: guest writes can
modify only its private copy, never the host packet.

## Semantic canonicality, not packet replay

The physical decoder and canonical encoder are distinct Wasm binaries covered
by a separately pinned, independently executed semantic audit. The decoder
emits a schema-bound semantic state containing complete quantizer decisions;
the audited schema forbids raw packet bytes. A separate zero-import canonical
encoder receives only that semantic state—never the packet or callback—and
must regenerate the byte-identical canonical packet.

The independent receipt must establish causal decision regeneration, complete
packet consumption, rejection of trailing data and malleable aliases,
decode-then-independent-encode correctness, and canonical uniqueness. This
closes the v3 loophole where a purported canonical re-encoder could simply
copy its input.

## Preserved scientific and physical gates

V4 delegates scientific provenance and final acceptance to the exactly pinned
v3 authority. One distinct packet remains mandatory for every routed expert.
Every claimed architecture family must independently satisfy:

- physical `R` in `[2.15, 2.5]` bpw;
- absolute `F <= 0.8`;
- page-rounded routed-expert cold read strictly below `2x`;
- positive Qwen-minus-strongest-control source-specific advantage of at least
  `0.03 bpw`;
- at least two non-aliased SwiGLU-MoE architecture families.

No shared stream or caller-supplied rate, distortion, provenance, or read
metric is accepted.

## Frozen source-only checks

```bash
python -I -B verify_source.py \
  --package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v4_wasm_authority \
  --expected-manifest-sha256 MANIFEST_SHA256

python -I -B run_source_gate.py \
  --package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v4_wasm_authority \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --v3-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v3_physical_authority \
  --review-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v3_physical_authority_independent_source_review_20260902 \
  --output /tmp/strata-rm-v4-source-gate.json
```

The source gate has no argument for model data, packets, runtime-audit
packages, semantic-decoder packages, or scientific evidence. Production stays
fail-closed until all independent external pins and the explicit authorization
`AUDIT_ROUTED_EXPERT_GLOBAL_RM_SWAP_RESULT_V4` are supplied.

```text
FROZEN_V4_WASM_AUTHORITY_SOURCE_ONLY__HOLD_RUNTIME_PAYLOAD_AND_RD
```
