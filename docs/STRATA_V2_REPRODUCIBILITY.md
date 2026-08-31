# Reproducing and auditing STRATA-XKLT-SC v2

> **Result status:** passed and published. Commands below distinguish the
> original blind execution from source-free release verification and later
> deterministic replay. The final values and hashes are in
> [STRATA_V2_RESULTS.md](STRATA_V2_RESULTS.md).

## Three different activities

Do not conflate these:

1. **Protocol verification** checks that selection and codec controls were
   sealed before source access.
2. **Deterministic replay** reruns the frozen algorithms on the same now-known
   sources. It can test implementation reproducibility, but it cannot recreate
   the original blindness.
3. **Artifact audit** independently parses and scores the canonical one-shot
   output. This is the normative way to verify the published blind result.

Once source access has occurred, no rerun—however bit-identical—becomes a new
blind confirmation. Retain the original create-only locks and output tree.

## Published compact layout

The repository preserves freeze-bound files at their canonical paths so the
sealed path/hash map remains directly checkable:

```text
strata_v2_codec/
  common.py
  emit_and_lock.py
  polar_encoder.py
  run_one_shot.py
  FORMAT.md
  test_common.py
  test_emitter_contract.py
  test_emitter_synthetic.py

blind_protocol_v2/
  audit_v1_result.py
  prepare_selection_proposal.py
  validate_proposal.py
  selection.proposal.lock.json
  route_table.proposal.bin
  route_table.proposal.audit.json
  unopened_snapshot.audit.json
  v1_failure_independent_audit.json
  build_codec_freeze_v2.py
  validate_codec_freeze_v2.py
  codec_freeze.lock.json
  codec_freeze.validation.json
  materialize_full_tensors_v2.py

bg_codec_bec_encoder.py
strata_v2_klt_mixed_independent_auditor_v1.py
strata_v2_klt_lineage_tamper_tests_v1.py
strata_v2_klt_known_n20_independent_validation_v1.py
strata_v2_klt_known_n20_fixture_v1.json

strata_v2_blind_one_shot_v2/
  strata_xklt_sc_v2.bin
  summary.json
  ONE_SHOT_INTENT.json
  preencoding_manifest.json
  allocation.lock.json
  EMISSION_RECEIPT.json
  independent_audit/{inspection.json,independent_decode_audit.json}
  independent_lineage_tamper_tests.json

release/
  strata_v2_release_manifest.json
  strata_v2_postrun_test_receipt.json

tools/verify_strata_v2_release.py
```

The compact release includes the source-finalization lock, pre-encoding
manifest, allocation lock, intent/summary, physical container, independent
inspection/audit JSON, and tamper receipt. Original Qwen BF16 payloads,
per-block payload duplicates, and FP64 reconstruction arrays are deliberately
excluded; the authenticated byte-range provenance, hashes, and frozen
materializer are sufficient for an authorized replay.

One historical dependency is intentionally not redistributed:

```text
agent_polaris_qwen_rht_encoder.py
SHA-256 062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0
```

Its own comments identify a direct port of the pinned upstream MATLAB code,
whose repository exposed no explicit license grant. The freeze and one-shot
intent retain the path/hash, while the release manifest declares it withheld
and the compact verifier checks that declaration. Source-free artifact
verification and the published independent audit remain complete. Exact
historical re-encoding requires a lawfully obtained byte-identical copy; this
repository cannot supply one or grant permission for it.

From a fresh checkout, the standard-library-only compact verifier needs no
Qwen payload and no CuPy installation:

```bash
python tools/verify_strata_v2_release.py --repo-root .
```

It checks every manifest byte/hash, frozen-artifact identities, internal
seals and cross-document bindings, physical framing/rate/padding, the one-shot
contract, pooled SSE/energy arithmetic, the independent primary gate, and all
ten rejected tamper cases.

## Frozen runtime

The canonical freeze binds more than package version strings:

| Component | Frozen value |
|---|---|
| Python | 3.12.3; executable SHA-256 `1d3cf64f...74fd5` |
| NumPy | 2.5.2; complete distribution tree bound |
| SciPy | 1.18.1; complete distribution tree bound |
| CuPy | `cupy-cuda12x==14.2.0`; complete distribution tree bound |
| cuda-pathfinder | 1.8.0; complete distribution tree bound |
| CuPy CUDA runtime | 12.9 (`12090`) |
| NVIDIA driver API | CUDA 13.0 (`13000`) |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0 |
| Canonical interpreter | `/workspace/int2-cupy-venv/bin/python` |
| Polar construction checkout | `/root/PolarLatticeQuantization` |

Blind-mode execution and audit compare the interpreter, wheel RECORD entries,
actual installed files, package-tree hashes, CUDA values, and device identity
with the freeze. A merely version-compatible environment is suitable for a
development replay, not for relabelling the canonical blind result.

Set paths used below:

```bash
export WS=/workspace/INT2__compression
export PY=/workspace/int2-cupy-venv/bin/python
export POLAR_REPO=/root/PolarLatticeQuantization
cd "$WS"
```

## Source-free self-tests

These do not need the selected v2 weights:

```bash
"$PY" -m unittest \
  strata_v2_codec.test_common \
  strata_v2_codec.test_emitter_contract

"$PY" strata_v2_codec/test_emitter_synthetic.py \
  --format strata_v2_codec/FORMAT.md

"$PY" strata_v2_klt_mixed_independent_auditor_v1.py --self-test

"$PY" strata_v2_klt_known_n20_independent_validation_v1.py \
  --fixture strata_v2_klt_known_n20_fixture_v1.json \
  --result PATH/TO/klt_n20_block13_result.json \
  --container PATH/TO/klt_n20_block13.polar.bin \
  --source-bf16 PATH/TO/block_13_n20.bf16.bin \
  --output /tmp/strata-v2-known-n20-audit.json \
  --inverse-device cupy
```

The known `N=2^20` development fixture should demonstrate a complete causal
decode/re-encode and reproduce its frozen source-code metric. It is
development evidence only, not part of the second-panel score. The result,
container, and BF16 source named in that optional command are not yet present
in the publication repository; verify their hashes against the fixture before
running it.

## Historical pre-access sequence

The following sequence describes how the original controls were created. It
must be run only in the corresponding pristine pre-access workspace state.
Running proposal/freeze builders after source access cannot prove prior
blindness and should be rejected by their state gates.

First independently rescore v1, generate the metadata-only proposal, and
validate it:

```bash
"$PY" blind_protocol_v2/audit_v1_result.py \
  --workspace "$WS" \
  --output "$WS/blind_protocol_v2/v1_failure_independent_audit.json"

"$PY" blind_protocol_v2/prepare_selection_proposal.py \
  --workspace "$WS" \
  --metadata-cache "$WS/qwen_weight_cache" \
  --output-dir "$WS/blind_protocol_v2"

"$PY" blind_protocol_v2/validate_proposal.py \
  --workspace "$WS" \
  --proposal-dir "$WS/blind_protocol_v2"
```

Then build the codec freeze from already-opened development evidence and
validate it before any materializer or selected source output exists:

```bash
"$PY" blind_protocol_v2/build_codec_freeze_v2.py \
  --workspace "$WS" \
  --development-run-dir "$WS/strata_v2_dev_final_exact_runtime_20260831a" \
  --development-audit \
    "$WS/strata_v2_dev_final_exact_runtime_20260831a/independent_audit/independent_decode_audit.json" \
  --development-tamper-audit \
    "$WS/strata_v2_dev_final_exact_runtime_20260831a/independent_lineage_tamper_tests.json" \
  --output "$WS/blind_protocol_v2/codec_freeze.lock.json"

"$PY" blind_protocol_v2/validate_codec_freeze_v2.py \
  --workspace "$WS" \
  --freeze "$WS/blind_protocol_v2/codec_freeze.lock.json" \
  --output "$WS/blind_protocol_v2/codec_freeze.validation.json"
```

Both output files are create-only. The validator records the current
pre-access state and intentionally refuses a post-access workspace.

## Deterministic source replay

The canonical materializer has no tensor, URL, range, or destination
override. After reviewing the Qwen license and understanding that a replay is
no longer a new blind experiment, the frozen invocation is:

```bash
"$PY" blind_protocol_v2/materialize_full_tensors_v2.py \
  --workspace "$WS" \
  --authorization-phrase \
    "AUTHORIZE SEALED QWEN V2 ONE-SHOT MATERIALIZATION"
```

It requires the exact proposal, freeze, and validation hashes, makes only the
eighteen sealed HTTP range requests, requires exact 206 receipts, writes via a
temporary create-only tree, and atomically finalizes
`blind_protocol_v2/unblinded/source_hashes.lock.json`. Never substitute a
whole shard or alternate tensor and retain the blind label.

## One-shot provenance command

The canonical blind process was launched with:

```bash
"$PY" strata_v2_codec/run_one_shot.py \
  --workspace "$WS" \
  --selection-lock blind_protocol_v2/selection.proposal.lock.json \
  --route blind_protocol_v2/route_table.proposal.bin \
  --source-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --protocol-mode blind \
  --codec-freeze blind_protocol_v2/codec_freeze.lock.json \
  --format strata_v2_codec/FORMAT.md \
  --independent-auditor strata_v2_klt_mixed_independent_auditor_v1.py \
  --output-dir strata_v2_blind_one_shot_v2 \
  --python "$PY" \
  --encoder strata_v2_codec/polar_encoder.py \
  --polar-repo "$POLAR_REPO" \
  --workers 7
```

This command is a historical provenance record. It cannot be rerun from the
compact public checkout alone because the hash-bound base encoder described
above is withheld.

The output directory must not exist. The runner writes a one-shot intent,
invokes exactly fourteen encoders, and has no resume code. If interrupted or
failed, preserve the directory as the result. Do not delete it and rerun under
the same claim.

For a non-confirmatory development rehearsal on other sources, use
`--protocol-mode development --allow-development-rehearsal` with a matching
development selection/source/freeze fixture. Development mode is permanently
ineligible for a positive blind claim.

## Independent inspection and source-domain audit

After the canonical summary and container exist, first run the independent
auditor's source-free parser if desired:

```bash
"$PY" strata_v2_klt_mixed_independent_auditor_v1.py \
  --container strata_v2_blind_one_shot_v2/strata_xklt_sc_v2.bin \
  --output-dir strata_v2_blind_one_shot_v2/independent_inspection \
  --inspect-only
```

The normative audit supplies the complete lineage atomically:

```bash
"$PY" strata_v2_klt_mixed_independent_auditor_v1.py \
  --container strata_v2_blind_one_shot_v2/strata_xklt_sc_v2.bin \
  --output-dir strata_v2_blind_one_shot_v2/independent_audit \
  --workers 7 \
  --inverse-device cupy \
  --protocol-mode blind \
  --selection-lock blind_protocol_v2/selection.proposal.lock.json \
  --source-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --codec-freeze blind_protocol_v2/codec_freeze.lock.json \
  --format-freeze strata_v2_codec/FORMAT.md \
  --preencoding-manifest strata_v2_blind_one_shot_v2/preencoding_manifest.json \
  --allocation-lock strata_v2_blind_one_shot_v2/allocation.lock.json \
  --one-shot-intent strata_v2_blind_one_shot_v2/ONE_SHOT_INTENT.json \
  --one-shot-summary strata_v2_blind_one_shot_v2/summary.json \
  --source-root blind_protocol_v2/unblinded
```

All source-lineage arguments are mandatory together. Supplying only some is
an error. In blind mode the auditor also requires the executing auditor and
complete runtime environment to match the freeze.

The audit emits large FP64 reconstruction arrays for traceability. The
normative compact output is
`independent_audit/independent_decode_audit.json`. Trust its
`primary_claim_gate`, not the encoder summary's staging metric.

## Tamper tests

Only after a complete audit should the lineage tamper harness run:

```bash
"$PY" strata_v2_klt_lineage_tamper_tests_v1.py \
  --container strata_v2_blind_one_shot_v2/strata_xklt_sc_v2.bin \
  --protocol-mode blind \
  --selection-lock blind_protocol_v2/selection.proposal.lock.json \
  --source-lock blind_protocol_v2/unblinded/source_hashes.lock.json \
  --codec-freeze blind_protocol_v2/codec_freeze.lock.json \
  --format-freeze strata_v2_codec/FORMAT.md \
  --preencoding-manifest strata_v2_blind_one_shot_v2/preencoding_manifest.json \
  --allocation-lock strata_v2_blind_one_shot_v2/allocation.lock.json \
  --one-shot-intent strata_v2_blind_one_shot_v2/ONE_SHOT_INTENT.json \
  --one-shot-summary strata_v2_blind_one_shot_v2/summary.json \
  --source-root blind_protocol_v2/unblinded \
  --device cupy \
  --run-root strata_v2_blind_one_shot_v2
```

The receipt must contain ten unique rejected mutations, each with a nonempty
failure reason. A tamper-test failure invalidates the blind claim even if the
numeric MSE is favorable.

## Final verification checklist

The published release completed this checklist:

- verify all published files against the freeze and summary hashes;
- confirm physical size 7,608,729 bytes and integer rate headroom four bits;
- confirm every logical stream and zero padding/tail check;
- confirm fourteen canonical decode/re-encode byte matches;
- confirm 18/18 matrix and 108/108 nested source hashes;
- confirm independent KLT, label, scale, seed, and allocation rederivation;
- confirm FP64 source energy and SSE sum to the published pooled MSE;
- confirm `audit_execution_passed` separately from the scientific gate;
- confirm all ten tamper cases were rejected; and
- preserve and publish a failure unchanged if any condition is false.

All conditions were true. The exact values and artifact hashes are recorded in
the final result card and mechanically checked by the compact release verifier.
