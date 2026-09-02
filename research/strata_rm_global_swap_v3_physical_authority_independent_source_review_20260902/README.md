# Independent source review: global STRATA RM swap v3 authority

Date: 2026-09-02

This review covers frozen producer manifest
`9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83`
and source root
`83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad`.

It is a benign source-only review. The producer was not modified, and no
network, model, checkpoint, packet, decoder module, Qwen payload, or control
payload was opened. `review_static.ps1` independently authenticates the flat
producer closure and checks the requested source mechanisms. The checked-in
receipt is not Wasmtime execution evidence.

The core v2 findings are repaired in source. `REVIEW_ASSESSMENT.md` records
three narrower residual limits: packet mutation is checked only after both
calls, the 1 GiB memory cap is checked rather than enforced during a call, and
byte-identical re-emission alone is not semantic canonicality. The mandatory
independent decoder audit must close those points before payload authority.

```text
PASS_V3_STATIC_AUTHORITY_REPAIRS__CONDITIONAL_ON_PINNED_WASMTIME_AND_DECODER_AUDIT_OF_TRANSIENT_MUTATION_MEMORY_AND_CANONICALITY__HOLD_PAYLOAD_AND_RD
```

