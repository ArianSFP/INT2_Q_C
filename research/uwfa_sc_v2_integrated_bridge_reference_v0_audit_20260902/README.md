# Independent audit of the UWFA-SC v2 integrated bridge reference

This package audits only the source-free reference package
`research/uwfa_sc_v2_integrated_bridge_reference_v0` at pinned
`SOURCE_MANIFEST.json` SHA-256
`51f158c7f82fad81bd2b15d30e6581a2847e0e436d98f085055b8d818bf43f31`.

It does not inspect or authorize Qwen weights, an existing quantized
container, extracted decisions, matched-Gaussian controls, or the main v2
producer.  A PASS from this audit means only that the reusable reference
bridge mechanisms are source-ready within the claim boundary stated in the
producer README.

`independent_audit.py` authenticates the complete producer inventory before
executing immutable in-memory copies of the pinned Python sources.  It runs
the producer's 17 tests and an independent adversarial suite covering:

- integer ranges, overflow, canonical ordering, non-overlap and padding;
- header CRC and normalized semantic root behavior;
- explicit canonical model rows and serialized-model-only causal decoding;
- literal arithmetic byte and logical-length re-encoding;
- the exact six-expert/fifteen-frame owner topology;
- rational physical rate, numerical `F`, exact page unions, and owner-share
  conservation;
- completion-last API behavior and its deliberately excluded bootstrap
  properties; and
- the synthetic semantic reconstruction callback boundary.

The audit also constructs accepted counterexamples for every deliberate or
accidental divergence from
`docs/UWFA_SC_V2_INTEGRATED_CONTAINER_SPEC.md`.  Those counterexamples are
reported as limitations, not hidden by the positive mechanism tests.

Run on a source-only machine with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 independent_audit.py \
  ../uwfa_sc_v2_integrated_bridge_reference_v0
```

`AUDIT_RESULT.json` and `AUDIT_MANIFEST.json` are written only after the
frozen audit has run on the provided RunPod and its outputs have been checked.

