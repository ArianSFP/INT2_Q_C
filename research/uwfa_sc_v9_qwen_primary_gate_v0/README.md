# UWFA-SC v9 primary-only Qwen runtime gate

This is a new nonpromoting runtime envelope around the byte-sealed UWFA-SC v8
science. It does not modify v8, its 150 candidates, its disjoint-component
holdout, its model serialization, its physical framing, or its CuPy kernels.
It exists because the completed Qwen v8 run correctly aborted before fit after
budgeting the primary source, four survivor shuffles, and the coordinate
diagnostic together. The exact primary itself is materially smaller and the
authenticated RTX 5090 preflight supports a bounded primary-only run.

The admitted workload is exactly 38,621,316,130 cell-symbol updates: all three
disjoint stream-owner component folds plus the final complete-panel fit and
exact score. Runtime admission uses the conservative throughput in the live,
sealed-v8-validated source-free preflight:

```text
c = 0.5 * measured_updates_per_second
T_kernel = 38,621,316,130 / c
admit iff 1,800,000 <= c <= 4,500,000 and T_kernel <= 21,600 seconds
```

The authenticated reference observation was `c=3,242,398.2106118356`, giving
`T_kernel=11,911.3426609967` seconds. This is explicitly a GPU kernel-work
projection, not total launch time. Panel decoding, host arithmetic coding,
standalone/routed causal decoding, canonical rebuilding, physical metrics and
publication are timed in the completed run but are not modeled by that
projection.

The runner records the original 286,625,070,746-update maximum and the
93,518,490,096-update coordinate diagnostic, but neither is admitted or run.
There is no CLI or call path for shuffles, coordinate diagnostics or controls.
Even if physical and held-out primary gates pass, the terminal status is
`PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED` and controls
remain unauthorized.

Before opening the Qwen artifact, the runner authenticates this package, the
repaired exploratory support, and the sealed v8 package; runs the exact
all-150 and representative source-free CuPy gates; validates both complete
receipts against independent GPU identity; checks the measured/conservative
throughput derivation and bounds; and seals a `SourceFreeReview` capability.
The only artifact opener verifies that capability before delegating to the
retained no-follow reader.

The repaired group-ordinal ABI remains `list[list[int]]`. Each value is an
exact built-in `int`; tuple rows are forbidden because they change NumPy from
one-axis advanced indexing to one index per array axis. The panel is causally
decoded once and the exact same object is reused for the second sealed panel
extraction request.

The exact Qwen stream/fold/update pins are evaluation-runner tamper checks.
They are not decoder identity inputs and are not serialized into `UWFCV8.bin`.

## Source-only hostile test

On the POSIX RunPod checkout, before any payload authorization:

```bash
/usr/bin/python3.12 -I -B \
  research/uwfa_sc_v9_qwen_primary_gate_v0/test_source_only.py
```

The tests use only source snapshots and synthetic objects. They verify import
inertness, authorization/review ordering, capability tamper rejection, exact
fold workloads, all-150 candidate closure, exact primary call scope, absence
of survivor/control entrypoints, nonpromotion, list-of-list integer semantics,
and one underlying panel decode.

## Exact primary launch

After independent review, from `/workspace/INT2_Q_C`:

```bash
/workspace/int2-cupy-venv/bin/python -I -B \
  research/uwfa_sc_v9_qwen_primary_gate_v0/primary_gate.py \
  --authorization RUN_EXACT_QWEN_PRIMARY_ONLY_NONPROMOTING_V0 \
  --v9-package /workspace/INT2_Q_C/research/uwfa_sc_v9_qwen_primary_gate_v0 \
  --pinned-support /workspace/INT2_Q_C/research/uwfa_sc_v8_qwen_early_gate_v0/early_gate.py \
  --v8-package /workspace/INT2_Q_C/research/unifilar_wfa_entropy_census_stage0_v8 \
  --strata-common /workspace/INT2_Q_C/strata_expert_local_codec/common.py \
  --frozen-auditor /workspace/INT2_Q_C/strata_v2_klt_mixed_independent_auditor_v1.py \
  --artifact /workspace/INT2__compression/strata_expert_affine_milestone_v1/strata_expert_affine_n20n21.bin \
  --output-dir /workspace/uwfa_sc_v9_qwen_primary_gate_v0_result
```

`COMPLETE.json` is written last. It is a content-completion seal for this
nonpromoting diagnostic, not positive publication authority.
