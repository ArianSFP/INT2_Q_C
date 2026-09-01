# BiSCo shallow raw-MSE oracle

Status: **implemented, preregistered, CPU-audited, and ready for a CuPy
launch when the shared GPU is released.**  No pinned Qwen matrix has been
opened by this branch and no result is claimed yet.

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

## RunPod launch (only after the GPU is released)

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
```

The JSON includes per-matrix sufficient statistics, per-expert folds, all
checkpoint curves, exact source and protocol hashes, paired initialization
hashes, both physical ledgers, model schemas, and artifact hashes.

## Independent verification

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

## CPU preflight

No CUDA context is created by the tests:

```bash
cd INT2_Q_C/research/bisco_raw_mse_oracle
/workspace/int2-cupy-venv/bin/python -m unittest -v test_bisco_raw_mse_oracle.py
```

The suite covers protocol bindings, exact independent ledgers, the strict file
firewall, charged FP16 normalization, paired initialization, a numerical
gradient check, monotone bit flips, whole-expert aggregation, deterministic
Gaussian generation, and both sides of the early-stop boundary.

## Claim boundary

A negative result rejects only the frozen shallow `d=16, h=64, 18+18` recipe.
It does not prove that arbitrary nonlinear decoders, deeper BiSCo variants, or
all learned weight priors cannot work.  A positive auxiliary result is also not
a final Qwen-panel result, because the gate-role training firewall remains
unresolved.  No perplexity, activation loss, distillation, protected-channel,
or uncharged decoder claim is accepted as a substitute for original-coordinate
source MSE.
