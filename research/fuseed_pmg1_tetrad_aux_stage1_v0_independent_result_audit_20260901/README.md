# Independent result audit: PMG1 tetrad auxiliary stage-1 v0

Verdict: **PASS** for the narrowly scoped auxiliary development-panel result.
This is not stage-2, Gate, validation, pinned-panel, codec, compression, or
target-achievement evidence.

The audit bound the exact three-file producer closure, the frozen PMG plan,
the source manifest, all 23 referenced Up/Down source files, and a new
disjoint RunPod replay. The replay result is structurally identical to the
producer result after deleting only `runtime.elapsed_seconds`; its independently
created file has SHA-256
`233f4063885cec71805bc30ad58d034b3b9c5fdae1c93bdd49bae23d0a0a779d`.

## Independent findings

The frozen plan regenerates exactly 2,048 fit and 2,048 score keys with empty
intersection. They span the expected 23 identities: twelve Down matrices and
eleven Up matrices, with expert 0/Up absent. Every row's fit/score cardinality
matches its regenerated identity count.

For every one of the 4,096 keys, the verifier independently parses the
canonical coordinate and checks the direct-counter map. Up uses
`native=(row+768)*2048+column`, offset `11520+16*(expert%32)`, and role 0;
Down uses `native=column*768+row`, offset `12032+8*(expert%32)`, and role 1.
Both then use `sequence=native%261120`, `quotient=native//261120`,
`lane=quotient&3`, `normal4=quotient>>2`, and addend
`1024+100*(expert//32)`. The inverse identity holds for every key.

All 23 five-word fits are reproduced by the disjoint GPU run. Independently,
the verifier decodes every recorded word as IEEE binary16, rejects negative
zero/nonfinite words, checks four coefficients plus intercept, and recomputes
every per-identity raw and centered capture. It also recomputes totals, both
role aggregates, all 23 delete-one values and jackknife standard error, all
16 control seeds and their mean/MC standard error, and all diagnostics.

The recomputed aggregate values are raw capture
`-0.04577526835279766`, centered capture `-0.03655885228971001`, and
delete-one three-SE upper bound `0.0015538337327504342`. Both role captures
are negative. The frozen promotion predicate is therefore false and the
reported `POLICY_REJECT_INCONCLUSIVE_FAR_SHORT_STOP_BEFORE_STAGE2` follows.
The planning threshold is explicitly non-converse; this result is not a
universal family impossibility claim.

Static inspection finds exactly two `fill_plan` calls, both for stage 1, and
no `reconstruct_plan` call. The result and replay each report 23 selection
Up/Down opens and zero Gate, old-validation, fresh-validation, pinned-panel,
and network access. The audit itself opened and hashed exactly those 23 source
artifacts on the authorized endpoint; it did not request stage-2 or pinned
data.

## Verification

The local checkout intentionally lacks the 23 source artifacts, so complete
verification must point at an already-open source root. On the authorized
RunPod, from this audit directory's parent, use:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B \
  fuseed_pmg1_tetrad_aux_stage1_v0_independent_result_audit_20260901/verify_audit.py \
  --audit-manifest-sha256 <EXTERNAL_PIN_FROM_RELEASE_HANDOFF> \
  --workspace /workspace/INT2__compression
```

The verifier opens every named artifact through a held regular-file
descriptor, hashes it, checks read stability and EOF, strictly parses JSON,
and enforces exact producer and audit-directory closure. The caller must
supply the externally recorded manifest SHA-256; there is no source-default
fallback. The receipt deliberately does not embed that pin because the
manifest hashes the receipt, which would create a circular digest.
