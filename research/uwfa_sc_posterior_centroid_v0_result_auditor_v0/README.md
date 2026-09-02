# UWFA-SC posterior-centroid v0 independent result auditor

Date: 2026-09-02

Status: frozen source-only verifier awaiting externally recorded result pins.
This package was built without opening a completed v9 publication, posterior
publication, BF16 score matrix, RunPod, CUDA device, or network resource.

The source snapshot root is SHA-256 over a fixed domain separator followed by
canonical exact member rows ordered by raw ASCII filename bytes. Locale- or
case-insensitive filesystem sorting is never part of the root definition.

## Why this package exists

`uwfa_sc_posterior_centroid_v0_final_independent_audit_20260902` is a useful
source audit, but it explicitly states that it never opened a completed v9
result, posterior output, or BF16 panel. It verifies source mechanics and
synthetic fixtures only. It is not a fail-closed audit of a real diagnostic
output.

This sibling supplies the missing result-audit boundary. A PASS requires an
externally pinned completed v9 publication, a successful independently frozen
v9 result-audit receipt for those exact seven bytes, the completed posterior
publication, the complete identity-free BF16 panel, and all exact decoder
sources.

## What is independently recomputed

The auditor does not import `diagnostic.py` or `posterior_core.py`, and the
posterior `RESULT.json` is never a numerical input. Its independent core:

1. authenticates the exact v9 and posterior directory membership, every
   member byte, external member pins, completion rows and internal seals;
2. authenticates the posterior producer closure only to retain its narrow
   coordinate-decoder bridge, plus the exact v8/STRATA decoder closure;
3. requires the separately frozen v9 result-audit receipt to bind the exact
   predecessor publication before any posterior claim is considered;
4. redecodes every causal SC decision, lattice index, state trace, owner set,
   scale, RHT seed and reconstruction from literal `UWFCV8.bin`;
5. independently parses and canonically rebuilds all nine fold wrappers and,
   when warranted, the final wrapper;
6. independently implements the local, state-aware and independently
   permuted laws, all eight ridge values, both inner directions per fold,
   refits, binary16 head serialization and exact head-byte comparison;
7. scores identity/local/state/permuted reconstruction against authenticated
   original BF16 matrices in the original Gate/Up/Down metric;
8. recomputes pooled scores, `Delta_s`, `G_state`, every fold gate, final
   score, literal bytes, exact rational rate, `F` and terminal decision;
9. reruns the authenticated v8 descriptor-backed routed inner decoder and
   independently projects the literal suffix-page request into descriptor,
   repeated-request and unique-request cold-read ledgers; and
10. compares all material producer fields against the recomputation, then
    publishes independent input/result manifests and a completion-last seal.

Every emitted binary16 head must be byte-identical to the independently
selected/refitted head. A JSON claim cannot substitute for a missing or
different wrapper.

## Mandatory external authority

`UNRESOLVED_EXTERNAL_PINS.json` is intentionally unusable. After both
publications exist, an external process must record:

- absolute immutable paths;
- exact bytes and SHA-256 for all seven v9 publication members;
- exact bytes and SHA-256 for all 11 posterior hard-kill members or all 12
  survivor members;
- SHA-256 of the successful v9 independent result-audit receipt;
- SHA-256 of the complete identity-free BF16 source manifest; and
- the frozen source hashes already listed in `design_lock.json`.

The external pin file itself must be canonical pretty JSON and its SHA-256
must be supplied independently on the command line. The auditor refuses an
existing output directory.

## Invocation

On the authenticated Linux/CuPy host, using absolute paths in the pin file:

```text
/workspace/int2-cupy-venv/bin/python -I -B \
  research/uwfa_sc_posterior_centroid_v0_result_auditor_v0/result_auditor.py \
  --authorization AUDIT_EXACT_UWFA_SC_POSTERIOR_CENTROID_V0_RESULT \
  --expected-auditor-source-manifest-sha256 <this SOURCE_MANIFEST SHA-256> \
  --external-pins /absolute/posterior-result-audit-pins.json \
  --expected-external-pins-sha256 <independently recorded SHA-256> \
  --rht-device cupy
```

The output contains `AUDIT_RESULT.json`, `INPUT_MANIFEST.json`, `LIMITS.json`
and completion-last `COMPLETE.json`.

## Hostile source-only test

```text
python -I -B \
  research/uwfa_sc_posterior_centroid_v0_result_auditor_v0/test_source_only.py

python -I -B \
  research/uwfa_sc_posterior_centroid_v0_result_auditor_v0/verify_source.py \
  --package /absolute/research/uwfa_sc_posterior_centroid_v0_result_auditor_v0
```

The tests use synthetic arrays and temporary files only. They reject duplicate
or nonfinite JSON, weakened source pins, positive-authority laundering,
unexpected members, wrapper/head tampering, source identity fields, over-2x
gates and backend import before authentication.

## Exact limitation: no inference-ready routed posterior application

Posterior v0 still has no routed decoder that accumulates occupancies and
applies `CAGEPC0` inside the same one-pass expert decode. The auditor executes
the actual authenticated v8 inner routed decode and independently charges one
literal suffix-page request, but the posterior reconstruction is scored by the
offline full-panel path. Consequently:

- `actual_posterior_wrapper_routed_decode_executed = false`;
- `posterior_head_applied_inside_routed_session = false`;
- all read numbers are nonpromoting projections; and
- this auditor cannot promote an MoE inference/read claim.

A future inference-ready decoder must reproduce the offline reconstruction for
one routed expert without rereading compressed expert bytes and must receive a
new independent result audit.

This v0 result also supplies no matched-Gaussian, structure-destruction,
portability, or disjoint SwiGLU-MoE evidence. Even a numerical PASS remains a
nonpromoting discovery result, not a universal codec result.
