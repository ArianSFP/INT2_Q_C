# BiSCo shallow raw-MSE oracle

Status: **executed on CuPy, independently replayed from serialized state, and
hard-killed at the preregistered update-512 gate.**  No pinned Qwen matrix was
opened by this branch.  The sealed outcome is a negative result for this exact
shallow cell, not a negative result for nonlinear quantization in general.

This is the cheapest favorable falsification test for the nonlinear implicit
binary codebook described in the BiSCo red-team assessment.  It is deliberately
an auxiliary-only gate: it can kill or preserve the fixed `d=16` shallow cell,
but it cannot authorize a pinned-panel run.

## Exact question

At physical rate `R`, define

```text
F = D_Qwen * 2^(2R)
s_absolute = -0.5 log2(F)
s_match = -0.5 log2(D_Qwen / D_Gaussian)
```

The eventual target is `F <= 0.8`, equivalently
`s_absolute >= 0.16096404744368115`, at `2.15 <= R <= 2.5`, with cold
per-expert compressed reads below `2x`.  This gate additionally measures the
source-specific advantage against an identically trained iid Gaussian codec.

The parent preregistration is
[`bisco_protocol_freeze.json`](../breakthrough_redteam/bisco_protocol_freeze.json)
and the architectural/rate argument is
[`BISCO_BSQ_ASSESSMENT.md`](../breakthrough_redteam/BISCO_BSQ_ASSESSMENT.md).
`launch_protocol.json` freezes every remaining implementation choice before an
auxiliary result exists.

## Sealed run_1 outcome

The frozen CuPy run stopped at update 512 with

| Statistic | Published FP32-reduction result | Independent FP64 replay |
|---|---:|---:|
| `D_Qwen` | 0.11020813815423404 | 0.11020813758494276 |
| `D_Gaussian` | 0.10961316220157559 | 0.10961316325045290 |
| `s_match` | -0.003904857950836688 | -0.003904847322141395 |

The published `upper_s_match_2se` is `-0.00133385222544977`.  Together
with the update-256 value, it gives a recent improvement of only
`0.0018524339032029444`; both frozen early-stop inequalities hold.  The exact
decision is therefore `HARD_KILL_D16_SHALLOW_BEFORE_PINNED`.

The state-backed replay regenerated all eight untouched Qwen validation
matrices and all eight filename-seeded matched-Gaussian controls from the 32
frozen auxiliary files.  Its independently written evaluator reproduced every
published FP32 matrix SSE exactly.  Replacing the published FP32 reduction
with FP64 accumulation changed any matrix SSE by at most
`3.3301501363830925e-08` relative; independently blocked FP64 source energies
differed by at most `1.2299459866810625e-15` relative.

The replay also proves byte-for-byte that each 18,048-byte FP16 two-role
decoder is the IEEE-binary16 rounding of the decoder fields in its 72,224-byte
FP32 training state.  It parses those states from a separately defined closed
schema and fail-closes on extra, missing, reordered, or inconsistent
history/decision fields.

Frozen bindings:

- result SHA-256: `5904e3887e69cf47ee4a882aeaacceb27823504c1e23eeff6adb4b3360874d92`;
- replay receipt SHA-256: `f75fc33b9b67cb3b711e2f54a95994757ed062364d980ad9849458e69feb76e7`;
- canonical unsigned receipt seal: `4ca302c0232640efa34b072baec50c74ec295046f53c57e65973b7212345c04b`;
- independent replay implementation SHA-256:
  `0a3c38fab4cbac640b1731e66b72e94fefe370b015a9cbe99d19a585114d0bd1`.

See [`run_1/independent_replay_receipt.json`](run_1/independent_replay_receipt.json)
for per-matrix code/reconstruction hashes, FP64 SSEs, normalization checks,
comparison errors, backend identity, input bindings, and the canonical seal.
The concise result-specific audit narrative is
[`run_1/INDEPENDENT_REPLAY.md`](run_1/INDEPENDENT_REPLAY.md).

## Fixed codec

Each canonical `768 x 2048` matrix is flattened row-major into 16-weight
chunks.  Down projections are transposed first, so all roles have the same
canonical geometry.  A matrix stores one FP16 mean and one FP16 centered RMS;
normalization, reconstruction, and final scoring use those serialized values.

For each available role (`up`, `down`), the experiment trains two stages:

```text
encoder: Linear(16,64) -> SiLU -> Linear(64,18) -> sign / sqrt(18)
decoder: Linear(18,64) -> SiLU -> Linear(64,16)

x_hat = decoder_1(q_1) + decoder_2(q_2)
q_2 encodes x - decoder_1(q_1)
```

The encoders exist only to infer stored bits.  The deployed object contains
only FP16 decoders.  Training uses hard signs in the forward pass, an annealed
tanh straight-through derivative, a fixed bit-balance penalty, and paired Adam
updates.  Evaluation first rounds every decoder parameter through FP16 and
then performs one deterministic greedy pass over all 36 charged bits.  A flip
is accepted independently per chunk only when it reduces squared error; unit
tests prove the implementation cannot increase any chunk's error.

## Data firewall and control

The executable accepts only one source argument, `--aux-dir`; there is no
target/pinned argument.  It rejects a directory unless its BF16 set is exactly

```text
l15e{0,8,...,120}_{up,down}.bf16.bin
```

and explicitly rejects paths under `blind_protocol`.  File hashes are bound
before optimization.  Experts `{24,56,88,120}` are complete untouched
validation folds; the other 12 experts train the models.  All validation
statistics pool complete matrices in original coordinates, first within a
whole expert and then across experts.

The Gaussian control is generated from pooled **training-role** mean and RMS.
It has the same matrix counts and shapes and follows the same charged FP16
normalization path.  For Qwen and Gaussian, all of the following are identical:

- initial parameter bits;
- batch indices and update count;
- architecture, optimizer, temperature, and regularizer;
- evaluation checkpoints and greedy bit order.

This pairing separates a Qwen-specific advantage from finite-dimensional
neural-codec and optimizer loss.  The output reports both `s_match` and the
Gaussian operational gap; they are never conflated.

## Exact physical and cold-read ledger

Although the auxiliary data expose only `up` and `down`, the ledger charges
six independent deployment decoders: two stages for each of
`gate/up/down`.  It never obtains an artificially cheap rate by omitting the
unavailable gate role.

| Amortization | Decoder bytes | Expert code bytes | Physical R | Cold expert bytes | Cold read amp | Ideal-Gaussian minimum matched `s` |
|---|---:|---:|---:|---:|---:|---:|
| Production, 128 experts | 27,072 | 1,327,104 | 2.250382317437066 | 1,354,444 | 1.020427859096027x | 0.161346364880747 |
| Self-contained, 6 experts | 27,072 | 1,327,104 | 2.257742422598380 | 1,354,444 | 1.017101325352715x | 0.168706470042061 |

The ledger includes a 256-byte global header and 12 local FP16 moment bytes per
expert.  Attributed bytes amortize the decoder/header; cold reads load the
complete decoder/header once.  Both rows independently satisfy the physical
rate interval and strict `<2x` read rule.

The trained auxiliary artifact contains only four decoders and is clearly
labelled as such.  It is evidence for the gate, not the deployment artifact;
the target calculations always use the full six-decoder ledger above.

## Aggressive preregistered early stop

The fixed budget is 2,048 paired updates.  Complete whole-expert validation is
run at updates 256 (12.5%), 512 (25%), and 2,048.  Let

```text
U_t = pooled s_match at t + 2 * whole-expert SE
delta = U_512 - U_256
```

The family stops at update 512 exactly when

```text
U_512 < 0.08  and  delta < 0.01.
```

There are six more intervals of the same length.  Even projecting the most
favorable boundary at a constant recent slope gives

```text
U_full < 0.08 + 6 * 0.01 = 0.14,
```

below both the production requirement `0.1613464` and the conservative
six-expert requirement `0.1687065`.  This is an aggressive, preregistered trend
kill, not a theorem: the exact claim is only that this fixed shallow training
recipe did not justify further compute.

If the branch does not fire the stop, it completes the budget.  Promotion
still requires all four whole-expert folds to be positive, an absolute
serialized `F <= 0.8`, and both rate/read gates.  Even then the decision is
only `AUXILIARY_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL`.

## Why survival cannot open the pinned panel

The frozen auxiliary cache contains no `gate` projection, but the parent
architecture specifies a separate decoder for `gate`, `up`, and `down`.
Reusing the up decoder for gate or training a gate decoder on the pinned panel
would violate the data firewall.  Therefore this executable categorically
keeps `pinned_panel.opened=false`.  A survival would require a new,
independently frozen gate-role training source and protocol before any target
access.

## Executed RunPod launch

From `/workspace/INT2__compression`:

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/bisco_raw_mse_oracle/bisco_raw_mse_oracle.py \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --output-dir /workspace/INT2__compression/bisco_raw_mse_d16_run_1 \
  --backend cupy
```

The output directory must not already exist.  The command has no knobs for
updates, width, bits, split, learning rate, or stopping; all are sealed in the
launch protocol.  NumPy execution is rejected unless the explicit
`--allow-cpu-test-backend` flag is supplied, and such an output is rejected by
the production verifier.

Expected result artifacts are:

```text
bisco_raw_mse_result.json
qwen_training_state.fp32.bin
gaussian_training_state.fp32.bin
qwen_aux_up_down_decoder.fp16.bin
gaussian_aux_up_down_decoder.fp16.bin
verification_receipt.json
independent_replay_receipt.json
```

The JSON includes per-matrix sufficient statistics, per-expert folds, all
checkpoint curves, exact source and protocol hashes, paired initialization
hashes, both physical ledgers, model schemas, and artifact hashes.

## Arithmetic and binding verification

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/bisco_raw_mse_oracle/verify_bisco_raw_mse.py \
  --result /workspace/INT2__compression/bisco_raw_mse_d16_run_1/bisco_raw_mse_result.json \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --receipt /workspace/INT2__compression/bisco_raw_mse_d16_run_1/verification_receipt.json
```

The verifier uses only the Python standard library.  It independently
recomputes both ledgers, every pooled/fold statistic, Gaussian and absolute
`F/s` identities, the early-stop decision, protocol/script bindings, artifact
sizes/hashes, and (when `--aux-dir` is given) all 32 auxiliary source hashes.
It rejects CPU-labelled results and any result claiming pinned access.

## Independent serialized-state replay

`independent_replay.py` shares no evaluator code with the training oracle and
does not import it.  A fresh RunPod replay is:

```bash
cd /workspace/INT2__compression/INT2_Q_C
/workspace/int2-cupy-venv/bin/python \
  research/bisco_raw_mse_oracle/independent_replay.py \
  --result /workspace/INT2__compression/bisco_raw_mse_d16_run_1/bisco_raw_mse_result.json \
  --aux-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --output /workspace/INT2__compression/bisco_raw_mse_d16_run_1/independent_replay_receipt_fresh.json \
  --backend cupy
```

The replay performs four distinct checks:

1. It requires the exact frozen result and artifact hashes, parses both FP32
   states from its own hard-coded shapes/offsets, and exactly links the FP16
   decoder byte streams to rounded state fields.
2. It checks the complete field sets and exactly recomputes every derived
   statistic at updates 256 and 512, the final-history identity, the complete
   early-kill object, and the top-level decision.
3. It rehashes the exact 32-file auxiliary set, independently recomputes
   training-role moments, reconstructs held-out Qwen normalization and seeded
   Gaussian controls, and evaluates stored update-512 states with CuPy.
4. It accumulates scaled reconstruction errors in FP64, also emulates the
   original FP32 reduction, compares both paths under recorded justified
   tolerances, and seals all evidence with canonical JSON SHA-256.

The packaged receipt can be checked without a GPU or source cache.  This
revalidates its seal, current replay-script hash, local result/artifact hashes,
decoder/state equality, exact history/decision, every receipt comparison, and
all per-expert/pooled replay arithmetic:

```bash
python research/bisco_raw_mse_oracle/independent_replay.py \
  --verify-receipt research/bisco_raw_mse_oracle/run_1/independent_replay_receipt.json
```

## Tests

No CUDA context is created by the tests:

```bash
cd /workspace/INT2__compression/INT2_Q_C/research/bisco_raw_mse_oracle
BISCO_RUN_DIR=/workspace/INT2__compression/bisco_raw_mse_d16_run_1 \
  /workspace/int2-cupy-venv/bin/python -m unittest -v \
  test_bisco_raw_mse_oracle.py test_independent_replay.py
```

The combined suites cover protocol bindings, exact independent ledgers, the strict file
firewall, charged FP16 normalization, paired initialization, a numerical
gradient check, monotone bit flips, whole-expert aggregation, deterministic
Gaussian generation, both sides of the early-stop boundary, closed state
schemas, decoder/state byte equality, history and decision exactness, BF16
orientation, receipt input rebinding, and receipt tampering.

## Claim boundary

A negative result rejects only the frozen shallow `d=16, h=64, 18+18` recipe.
It does not prove that arbitrary nonlinear decoders, deeper BiSCo variants, or
all learned weight priors cannot work.  A positive auxiliary result is also not
a final Qwen-panel result, because the gate-role training firewall remains
unresolved.  No perplexity, activation loss, distillation, protected-channel,
or uncharged decoder claim is accepted as a substitute for original-coordinate
source MSE.
