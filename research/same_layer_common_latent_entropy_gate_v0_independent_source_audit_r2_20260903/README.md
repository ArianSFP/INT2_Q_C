# Independent repaired-source audit r2

This is a fresh, payload-blind review of producer manifest
`b92d4b5f307ba1d2b6bc6370d0b7cd118c4ab138dc6c8943402efe632a2a5d8f`
and source root
`f9fe8b64b31edc7599e8e9c302b7e283b2aed9cc24c165916ae3447a9f78311c`.

Verdict: `PASS_REPAIRED_SOURCE_ELIGIBLE_FOR_SEPARATE_DEPLOYMENT_REVIEW`.

The earlier negative audit remains correct for its superseded snapshot. This
review independently verifies the repair: only a family having at least one of
the exact endpoints `2.15` or `2.5` with the expected success status,
`capacity_ok=true`, and `strictly_below_2x=true` can enter charged-MDL/control
promotion. A Qwen-shaped source-free regression with excellent entropy gains
and four failed envelopes now produces a noneligible physical HOLD and invokes
zero controls. A second regression proves charged-MDL failure also skips
controls, while a single explicitly feasible family/endpoint runs exactly all
eight controls before it may survive.

The review also repeats manifest closure, HOLD ordering, all 32 Qwen binding
pins, quantizer, entropy/MDL, threshold, scramble, and local RTX 3060 CPU/CuPy
parity checks. No Qwen payload was opened and this is not payload execution
authority, a finite codec, or target evidence.
