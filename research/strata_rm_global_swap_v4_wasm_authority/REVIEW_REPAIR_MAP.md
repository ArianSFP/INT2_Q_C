# V3 review to v4 repair map

Pinned producer: v3 source root
`83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad`.

Pinned review: source root
`3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0`.

| V3 independent-review limit | Final v4 repair | Frozen hostile checks |
|---|---|---|
| Python `wasmtime`, native engine, and module provenance were inherited from the host. | Require an independently audited exact distribution closure. Pin manifest, audit-source root, receipt, capability, complete tree, Python modules, metadata version, native-library root, target, ABI, platform, and runtime version. Reauthenticate before import and mapped-native use. | 03–08, 18 |
| The 1 GiB memory check occurred after guest calls and there was no fuel budget. | Enable fuel on `Config`; call `Store.set_limits(...)` and `Store.set_fuel(...)` before either guest is instantiated. Require independent limiter and exhaustion probes. | 03, 05, 18 |
| V3 placed packet bytes in writable guest memory and detected only final-state mutation. | Retain the packet as immutable host-owned `bytes`. Expose only a bounded read callback. Reject overlapping requests and require an exactly-once partition of literal bytes. Guest never receives packet path, fd, pointer, WASI, or native I/O. | 16–19 |
| Byte replay did not prove the decoder understood a canonical code. | Pin distinct semantic decoder and independent zero-import canonical encoder. Audit complete decision recovery, a no-raw-packet semantic schema, causal regeneration, trailing-data and alias rejection, decode→encode equality, and uniqueness. Encoder receives semantic state only. | 09–14, 19 |

Preserved from v3 without relaxation: independent scientific-provenance audit,
cross-family alias rejection, exactly one packet per routed expert, literal
4 KiB page ledger, `R in [2.15,2.5]`, `F<=0.8`, `<2x` cold reads for every
family, strongest-control subtraction, absolute Qwen `F<=0.8`, and minimum two
architecture families.

No Wasmtime guest, checkpoint, tensor, packet, or model payload was opened in
constructing this source-only package.
