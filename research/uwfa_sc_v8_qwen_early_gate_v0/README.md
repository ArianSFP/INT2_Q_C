# UWFA-SC v8 Qwen early gate v0

This directory supplies one bounded early-kill diagnostic for the exact sealed
UWFA-SC v8 source. It does not authorize or make a positive compression claim.
The runner authenticates every v8 manifest member, executes the unchanged v8
all-150 selection, nested dependence-component holdout, literal serializer,
standalone decoder, canonical rebuild, routed descriptor reader, fixed STRATA
adapter and CuPy backend.

The Qwen artifact is not touched until the exact source-free all-150 and
representative GPU gates have passed. The runner then reconstructs the panel
identity from the artifact and constructs the strict v8 score receipt from the
already audited values:

```text
D      = 0.030902167403153148
SSE    = 500.39553685426534
energy = 16192.89450885593
```

The result includes literal physical bytes, rate, `F`, the nested winner,
per-component savings, descriptor-backed page reads, requested bytes both with
repetition and after interval coalescing, causal decode/re-encode evidence,
telemetry, and exact source hashes. Controls are never opened. Even if every
source gate survives, the only status is
`EARLY_DIAGNOSTIC_SOURCE_SURVIVOR_REQUIRES_CONTROLS_AND_INDEPENDENT_AUDIT`.

The pinned STRATA helper emits NumPy integer scalars for group ordinals while
the sealed v8 adapter requires exact built-in Python integers.  This runner
therefore applies one explicit value- and order-preserving `int()` conversion
to those ordinals.  The bridge receipt is included in the decoder-bundle hash.
This makes the run useful as an exploratory Qwen early-kill diagnostic, but it
is not execution of the sealed producer unchanged; a production result needs
a freshly reviewed producer revision with the ABI repair in its sealed source.

The sealed v8 source closure and external STRATA members are hash-pinned. The
score receipt is locally constructed from the fixed audited D/SSE/energy and
recomputed identities; the decoder-bundle, exploratory bootstrap and pipeline
bindings are also locally constructed and explicitly disclosed in the result.
They are not a substitute for the externally pinned production dispatcher.

## Source-only test

From this directory on a POSIX host:

```bash
/usr/bin/python3.12 -I -B test_source_only.py
```

This test authenticates the sealed source and exercises only synthetic bytes.
It never discovers or opens Qwen/model/control payloads and initializes no
CUDA context.

## Exact RunPod launch

From a clean checkout at `/workspace/INT2_Q_C`, with an absent output name:

```bash
cd /workspace/INT2_Q_C
/workspace/int2-cupy-venv/bin/python -I -B \
  research/uwfa_sc_v8_qwen_early_gate_v0/early_gate.py \
  --authorization RUN_EXACT_QWEN_EARLY_KILL_NO_CONTROLS_NO_CLAIM_V0 \
  --v8-package /workspace/INT2_Q_C/research/unifilar_wfa_entropy_census_stage0_v8 \
  --strata-common /workspace/INT2_Q_C/strata_expert_local_codec/common.py \
  --frozen-auditor /workspace/INT2_Q_C/strata_v2_klt_mixed_independent_auditor_v1.py \
  --artifact /workspace/INT2__compression/strata_expert_affine_milestone_v1/strata_expert_affine_n20n21.bin \
  --output-dir /workspace/uwfa_sc_v8_qwen_early_gate_v0_result
```

The output directory is created exclusively. `COMPLETE.json` is written last,
but it remains an early-diagnostic content seal rather than the v8 production
parent-marker authority.
