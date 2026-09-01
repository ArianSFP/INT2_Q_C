# Tier-A initialization-anchor oracle

This package asks one sharply bounded question: do any **publicly plausible,
source-decodable Hugging Face v4.51 initialization streams** still predict a
large enough component of final Qwen3 MoE expert weights to matter at the
project's 20%-below-Gaussian target?

It is a strict PTQ auxiliary gate, not a codec and not evidence about Qwen's
undisclosed training pipeline.  The immutable candidate lock was sealed before
any auxiliary weight payload was opened.

## Claim boundary

Public evidence fixes the model commit, config, normal initializer family and
scale (`mean=0`, `std=0.02`), and the Transformers v4.51 module structure.  It
does **not** disclose Qwen's production seed, framework, RNG implementation,
construction order, stream offsets, or parallel layout.  Consequently:

- a positive result is only an auxiliary survivor; it still needs a separately
  frozen gate-role experiment and a full physical-codec replay;
- a negative result rejects only the 56 locked Tier-A hypotheses;
- no winner may be called the actual Qwen initializer without independent
  documentation;
- safetensors serialization order is never treated as initialization order.

Speculative Megatron/parallel-layout emulations are deliberately deferred to a
separate lock.  Mixing them into this first gate would weaken the exact-source
claim and delay the decisive, documented-framework result.

## Immutable experiment

`candidate_lock.json` is protected twice:

- exact file SHA-256:
  `8f5ad9cb3bff21893e9fc2daf287942a43610f0af288e5070f173c93f05fb6ca`
- placeholder-normalized internal seal:
  `b7f100835e366c7ca68c189206dd953fbba7b737f00a19cacf74af277b131dbd`

The Cartesian candidate set is fixed at:

- 4 HF v4.51 stream scopes: tensor reset, layer reset, one global `post_init`
  stream, and constructor-consumption followed by global `post_init`;
- common seeds `0, 1, 42, 1234, 2023, 2024, 3407`;
- `float32 -> bfloat16` and direct `bfloat16` normal kernels;
- 56 candidates in the exact lock order, with one global candidate per search.

There is no arbitrary seed/offset search, per-tensor winner selection, retry,
or scientific command-line knob.

## Data firewall

The source manifest, source freeze, and held-out exclusion manifest are bound
by SHA-256.  A small packaged lock additionally freezes the exact one-tensor
intersection needed by this gate; the full exclusion manifest is revalidated
when present, while its absence cannot relax the packaged exclusion.  The
source directory must contain exactly the 32 expected regular,
non-symlink BF16 files.  The one tensor intersecting the held-out manifest,
`model.layers.15.mlp.experts.0.up_proj.weight`, is checked only as directory
metadata and is never opened or hashed by the runner or verifier.

The remaining 31 matrices are split by whole expert:

- candidate selection: 23 matrices from experts
  `0,8,16,32,40,48,64,72,80,96,104,112` (expert 0 has only `down` after the
  exclusion);
- untouched validation: both roles of experts `24,56,88,120`.

Execution order is fail-closed:

1. validate manifests and directory metadata;
2. pass CUDA Philox and PyTorch/CuPy interop parity without opening a source
   payload;
3. open/hash only candidate-selection payloads and fix the source, Gaussian,
   and permuted-control global winners;
4. only then open/hash the eight validation payloads.

The result includes an ordered access log, and the independent verifier
reconstructs that exact log.

## Sampling, fitting, and controls

Exactly 65,536 canonical coordinates are allocated over the 31 eligible
matrices by quotient/remainder: 32,768 fit and 32,768 score coordinates.
Coordinates come from revision/tensor/split-domain-separated SHA-256 rejection
sampling and are fit/score disjoint.  Down-projection coordinates are mapped to
their native `[2048,768]` storage only for sparse reads.

For every matrix and candidate, only fit coordinates estimate

```text
w_hat = mu + alpha * g
```

by float64 OLS with an intercept.  Score MSE uses the serialized fit-only
parameters.  Its denominator is the score error of the fit-only source-mean
predictor.  Score Pearson correlation is descriptive and never refits the
model.

Two matched searches use the same 56 candidates and tie-break:

- a stateless SHA-256/Box-Muller Gaussian source matched to each matrix's
  fit-only mean and centered RMS;
- a fixed SHA-256 permutation of initializer anchors, independently within
  matrix and split.

Validation capture is bias-corrected by subtracting the larger of zero and the
two validation control captures.  Uncertainty is the standard error across
four complete held-out experts, and up/down role folds are also reported.

## CUDA exactness and CuPy work

PyTorch CUDA `normal_` with an explicit `torch.Generator` is authoritative for
the locked Philox streams.  Before source access the runner:

- checks calculated generator offset increments against `Generator.get_offset`
  for five locked tensor lengths and both dtype paths;
- checks sequential generation against an explicit `set_offset` jump bitwise;
- checks PyTorch-to-CuPy DLPack values bitwise after exact float32 widening.

Any unavailable API or mismatch aborts before payload access.  CuPy performs
batched device-side staging and float64 sufficient-statistic evaluation;
CuPy's default RNG is never used.

## Physical/read target

The proposed source decoder has no learned generator table and reads zero
external generator bytes.  Per matrix it charges 23 bytes for family, seed,
offset, affine parameters, and layout flags: 414 bytes over the 18 target
matrices, or about `0.00011698405 bpw`.  A separate 4 KiB self-contained
descriptor sensitivity is reported.

The metadata charge is applied to both target inequalities.  Starting from the
current worst cold-read amplification of `1.169444x`, appending 69 bytes per
expert remains far below the strict `<2x` gate.

Decision states are frozen:

- hard kill when bias-corrected validation capture plus two whole-expert SE is
  below the metadata-adjusted composite requirement;
- standalone auxiliary survivor only when capture minus two SE reaches the
  metadata-adjusted current-result requirement, every whole-expert raw fold is
  positive, and the read gate passes;
- otherwise composite-only/inconclusive.

## RunPod commands

From the repository package directory, run the parity-free preflight and tests
in a process that has not imported torch or CuPy:

```bash
cd /workspace/INT2__compression/INT2_Q_C/research/initialization_anchor_oracle
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python -m unittest -v \
  test_initialization_anchor_gate.py

PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python initialization_anchor_gate.py \
  --preflight --workspace-root /workspace/INT2__compression
```

Production (the output directory must not already exist):

```bash
cd /workspace/INT2__compression
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/initialization_anchor_oracle/initialization_anchor_gate.py \
  --workspace-root /workspace/INT2__compression \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --output-dir /workspace/INT2__compression/init_anchor_aux_gate_v1 \
  --backend cupy
```

Independent verification, including rehashing all 31 eligible payloads while
leaving the excluded payload unopened:

```bash
cd /workspace/INT2__compression
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/initialization_anchor_oracle/verify_result.py \
  init_anchor_aux_gate_v1/initialization_anchor_result.json \
  --workspace-root /workspace/INT2__compression \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --receipt /workspace/INT2__compression/init_anchor_aux_gate_v1/verification_receipt.json
```

The production verifier rejects CPU results, altered locks/code/manifests,
non-finite JSON, winner/tie-break inconsistencies, fit leakage, broken fold or
target algebra, source-access reordering, and any claim that the pinned panel
was opened.
