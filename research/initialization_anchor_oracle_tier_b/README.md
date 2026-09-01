# Tier-B initialization-anchor oracle

## Outcome

This sealed strict-PTQ experiment **hard-kills the preregistered Tier-B procedural family**.  Searching 37,748,736 logical Megatron-Core/PyTorch-CUDA Philox keys did not discover a useful Qwen3 expert-weight anchor.

| Quantity | Result |
|---|---:|
| Raw source validation capture | `-0.0007326564576186634` |
| Largest of 32 matched null winners | `+0.0005681810954186739` (`scramble_11`) |
| Max-search-corrected capture, `C*` | `-0.0013008375530373373` |
| Whole-expert standard error | `0.000972032501989997` |
| Corrected upper `3 SE` bound | `+0.0016152599529326538` |
| Composite promotion threshold | `0.1456888483858212` |
| Empirical randomization p upper | `0.5454545454545454` |
| Decision | `HARD_KILL_BOUNDED_TIER_B_PROCEDURAL_SET` |

The upper confidence bound is about 90 times smaller than the composite threshold.  Experts 88 and 120 have negative source folds, both role folds are negative, and the source winner does not beat the 32 null winners.  This is a decisive negative result, not an inconclusive near miss.

The result says only that this finite procedural family failed.  It does **not** establish Qwen's initializer, training framework, seed, construction order, or lineage.  No primary source says that Qwen used the pinned Megatron-Core revision.  Tier A and its artifacts are unchanged.

## Exact hypothesis tested

The immutable protocol is [candidate_lock.json](candidate_lock.json).  It tests an expert-only persistent CUDA RNG stream with end-to-end seed

```text
CLI base_seed + 100*PP_rank + 1024 + 100*EP_rank + ETP_rank
```

This combines the pipeline-rank adjustment in Megatron-Core [`initialize.py`](https://github.com/NVIDIA/Megatron-LM/blob/1cb3264479f28b8526db3d335faa9c5ef2183989/megatron/training/initialize.py#L423-L450) with the expert-stream adjustment in [`random.py`](https://github.com/NVIDIA/Megatron-LM/blob/1cb3264479f28b8526db3d335faa9c5ef2183989/megatron/core/tensor_parallel/random.py#L406-L476), both pinned to commit `1cb3264479f28b8526db3d335faa9c5ef2183989`.

The logical key axes are:

- base seed `0..65535`;
- PP size `{1,2,4,8}`;
- EP size `{1,2,4,8,16,32,64,128}`;
- expert tensor-parallel size `{1,2,4}`;
- contiguous or round-robin expert assignment;
- separate gate/up/down, fused gate-up/down, or fused up-gate/down packing.

That is `65,536 × 4 × 8 × 3 × 2 × 3 = 37,748,736` logical ordinals.  For global layer 15 of a 48-layer model, PP1 and PP2 are exactly equivalent: both use PP rank 0 and local layer 15.  PP4 uses rank 1/local layer 3, and PP8 uses rank 2/local layer 3.  The frozen equivalence map therefore computes three PP representatives while retaining all four logical ordinals and the smallest-ordinal tie rule.  It evaluates 28,311,552 effective anchors.

### MCore call-ABI boundary

Tier B is an **expert-major, per-local-expert call ABI**:

```text
for local layer:
  for local expert:
    initialize FC1, then FC2
```

Its fused cases draw one `[2*(768/ETP), 2048]` FC1 tensor followed by one `[2048, 768/ETP]` FC2 tensor per expert, with both gate/up half interpretations.  This matches the construction shape/order hypothesis suggested by MCore `SequentialMLP`, whose pinned [`experts.py`](https://github.com/NVIDIA/Megatron-LM/blob/1cb3264479f28b8526db3d335faa9c5ef2183989/megatron/core/transformer/moe/experts.py#L1377-L1416) builds one `MLP` inside a local-expert loop.  The separate case instead makes three per-expert calls for gate, up, and down.

Tier B does **not** contain nontrivial `TEGroupedMLP` initialization.  The pinned grouped implementation invokes its FC1 builder once with `num_local_experts`, then its FC2 builder once.  Two missing ABIs are consequently separate hypotheses:

1. per-weight TE parameters initialized projection-major (`FC1 expert 0..E-1`, then `FC2 expert 0..E-1`);
2. one `single_grouped_weight` tensor/call for all local-expert FC1 weights, then one for all FC2 weights.

CUDA normal mapping depends on the call's total `numel`, so a grouped call is not equivalent to reshaping the Tier-B output.  Only the degenerate one-local-expert case can coincide conditionally.  These missing grouped ABIs require a separately sealed Tier-C protocol; they must not be retrofitted into this result.

## Source firewall

The experiment uses only the same 32-file layer-15 auxiliary cohort as Tier A.  The frozen held-out intersection excludes:

```text
model.layers.15.mlp.experts.0.up_proj.weight
```

The excluded payload was never opened.  The selection cohort contains 23 matrices from 12 experts.  Eight matrices from whole experts `{24,56,88,120}` remained closed until all 33 domain winners were persisted and SHA-bound.  The pinned evaluation panel was forbidden and remained unopened.

The result's access log is exact:

1. production parity before manifest, directory, or payload access;
2. 23 selection payload opens with immediate SHA verification;
3. a state-backed 33-winner freeze;
4. eight validation payload opens with immediate SHA verification.

There are 33 matched search domains: the source, 16 independent Gaussian pseudo-sources, and 16 independent within-matrix permutation/Rademacher scrambles.  Every null searches the identical deduplicated key family and cascade.  The decision subtracts `max(0, max(null validation capture))` from source validation capture, thereby charging the max-over-32 search.

## Frozen cascade

Stage 0 evaluates each 256-seed shard on 256 fit and 256 score coordinates.  It fits two role-global affine models and retains the exact lexicographic top 2,048 candidates per domain.  This generates at most 14,495,514,624 normal values.

The union of the 33 stage-0 lists contains 67,437 candidates, below the frozen cap of 67,584.  Stage 1 evaluates the same union for every domain on 48,624 selection coordinates, uses independent per-matrix fit-only affine models, and freezes one exact `(q, logical ordinal)` winner per domain.  Only then may validation open.

Each matrix's `mu` and `alpha` use fit coordinates only.  MSE, `q`, capture, and correlation use disjoint score coordinates.  Promotion also required the source to beat all null winners, every whole-expert fold to be positive, both role folds to be positive, and read amplification to remain below `2x`.

## CUDA parity and source-free calibration

The production generator is a CuPy C++17 `RawKernel` using `curand_init` and `curand_normal4`.  Before any payload access, it passed exact PyTorch-CUDA parity for:

- 45 descriptor cases covering float32, direct BF16, float32-to-BF16, five offsets through `4,294,967,300`, and all ETP shapes;
- 810 candidate-coordinate cases spanning PP classes, EP assignments/ranks, ETP shard boundaries, both roles, and all packings;
- nine persistent packing-offset cases;
- same-device PyTorch/CuPy DLPack.

The first source-free compile attempt failed closed because CUDA 12.8's libcu++ rejects C++14.  It produced no calibration JSON and opened no manifest or payload.  The sole change was the kernel compilation standard to C++17, after which CPU tests were rerun and the source-free calibration passed.

Calibration used 110,592 candidates × 512 synthetic coordinates = 56,623,104 values per repetition.  Its three measured kernel-only rates were approximately 58.584, 54.402, and 57.765 billion values/s; median `57,765,473,228.63944` values/s.  The estimate intentionally excludes source decode, null construction, and float64 reductions.

## Interruption and append-only recovery

The authorized run was deliberately paused after its first checkpoint audit.  Shards completed at about six per second, so 136 complete shards were journaled before `SIGTERM` took effect.  `np.savez_compressed` had also created an unjournaled 192,751-byte partial `stage0_136.npz`.

The journal had exactly 137 valid events: the immutable header plus shards `000..135`.  No event referenced the partial.  Python reported `BadZipFile: File is not a zip file`.  Rather than delete evidence, the exact file was moved unchanged to:

```text
interrupted_orphans/stage0_136.partial.sha256-7bb36fd08dc69647b50af44b801c2be1d4578da49c9e864cbf591c748016cd04.npz
```

Its size and hash are recorded in [recovery_orphan_stage0_136.json](recovery_orphan_stage0_136.json).  The completed journal was revalidated, then `--resume` restarted exactly at shard 136 and created a new, valid, journal-bound `state/files/stage0_136.npz`; the preserved partial remains under `interrupted_orphans/`.  The final journal has 392 SHA-chained events: header, 256 stage-0 shards, stage-0 merge, 132 stage-1 batches, stage-1 winners, and the validation firewall freeze.

## Physical accounting

The hypothetical surviving decoder would need one 8-byte global lineage descriptor plus 18 four-byte per-matrix affine pairs, 80 bytes total across 28,311,552 tested weights:

- side cost: `0.000022605613425925925 bpw`;
- metadata read: 20 bytes per requested expert;
- conservative cold-read amplification: `1.1694575633680555x`;
- external generator table/read: zero.

This passes the `<2x` read gate.  It does not rescue the scientific result because measured capture is negative.

## Artifact inventory

| Artifact | SHA-256 |
|---|---|
| `candidate_lock.json` | `bd1376d9bf4b13620d4a7c6c48a24cecd82d0054fa6899f4b343bad1ace23f23` |
| `tier_b_gate.py` | `83eb7682c8185d8f27dbd4b7d39de96cb54dad1c887a4cf026ac4ea759159665` |
| `common.py` | `75d8bbd7af9271ea5d2f099e7d720c1560bcc72864ac88f458095647468e7da3` |
| `kernels.py` | `b563b977251dd754f1d6ed7dfe08a486ae4ed6aab3ff60b1e9f9399be804a195` |
| `tier_b_source_free_calibration.json` | `92710a3c73533512f38f05b6010f4f59f307d308bced053411c51c8f4fbd1b23` |
| `result.json` | `e450c10767b54c190f901df8460c6ac57fe86cfaca7719c3db48475d4196fb92` |
| `recovery_orphan_stage0_136.json` | `075cdaa3feeb518f0a42336c946cbfe29cc2a4fee5cc84ca39b40395bb6f95b1` |
| final journal event | `2a1632c5735a1616af4fc6aab4997a1c46072af401c6422968d7385003dcf16a` |

The complete checksum inventory also binds the verifier, tests, documentation, and receipts in `ARTIFACT_SHA256SUMS.txt`.

## Verification

The independent verifier imports neither PyTorch nor CuPy.  Its basic mode recomputes protocol/source bindings, coordinate plans, candidate identities, affine algebra, all 33 fold trees, null correction, physical ledger, firewall ordering, embedded journal chain, calibration accounting, parity structure, and package hashes.  The exclusion firewall distinguishes immutable facts (both manifest hashes and the excluded tensor identity) from the environment-dependent record of whether the optional full external manifest was present.  It exact-compares the immutable facts, fail-closed validates the full manifest whenever present, and reports production-time and verification-time presence separately; a presence change alone is not a scientific mismatch.

```bash
cd /workspace/INT2__compression/INT2_Q_C/research/initialization_anchor_oracle_tier_b
PYTHONPATH=. /workspace/int2-cupy-venv/bin/python -m unittest -v \
  test_tier_b_gate.py test_verify_tier_b_result.py

PYTHONPATH=. /workspace/int2-cupy-venv/bin/python verify_tier_b_result.py result.json \
  --workspace-root /workspace/INT2__compression
```

Full forensic mode additionally rehashes every event and target, rejects extra state files, rebuilds all 33 stage-0 top-k merges, rebuilds the union and 33 stage-1 winners, checks the frozen winner JSON, verifies the preserved invalid orphan, and hashes all 31 eligible payloads without opening the exclusion:

```bash
PYTHONPATH=. /workspace/int2-cupy-venv/bin/python verify_tier_b_result.py result.json \
  --workspace-root /workspace/INT2__compression \
  --calibration tier_b_source_free_calibration.json \
  --recovery recovery_orphan_stage0_136.json \
  --output-dir /workspace/INT2__compression/tier_b_initialization_anchor_run_v1 \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --receipt verification_receipt_full.json
```

The verifier creates receipts with exclusive-create semantics and will not overwrite an existing receipt.

## Production command (historical)

This command is recorded for audit, not as an invitation to rerun the completed search:

```bash
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python \
  /workspace/INT2__compression/INT2_Q_C/research/initialization_anchor_oracle_tier_b/tier_b_gate.py run \
  --workspace-root /workspace/INT2__compression \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --output-dir /workspace/INT2__compression/tier_b_initialization_anchor_run_v1 \
  --calibration /workspace/INT2__compression/tier_b_source_free_calibration_v1.json
```

The resumed invocation added only `--resume`.  Scientific CLI knobs do not exist.
