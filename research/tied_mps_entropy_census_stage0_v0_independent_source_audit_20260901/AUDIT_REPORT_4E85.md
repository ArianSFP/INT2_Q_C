# Independent source audit: tied persistent-regime HMM census

## Outcome

**BLOCK — no payload authority.** The replacement source-manifest SHA-256
`4e85f7f4890fcb17aa967590fcad2c5ee0bab195c5b301f6c41eca0766618de0`
has exact source closure, and 13 selected native source-only tests pass on the
provided RunPod. Eleven correctness, lifecycle, or feasibility defects prevent
an authenticated numeric run.

The earlier manifest `a469...` failed closure and remains separately recorded
in `A469_BLOCK.json`. The later seal does not erase that event.

## Checks that passed

- Exactly seven declared regular source members plus `SOURCE_MANIFEST.json`
  were present. Every declared byte count and SHA-256 matched the `4e85...`
  manifest, with no extra package member.
- The standalone lossless threshold is correctly fixed at
  `0.1528899669629145 bpw`; the speculative `0.11356063457 bpw` gap is not used.
- The parameter ledger charges `pi`, dense `T`, context/state emissions, and a
  256-byte model header. The packet ledger charges a global header, immutable
  global bytes, model bytes, 64 bytes per stream, expert headers, payload byte
  padding, the 2.15-bpw floor, global page rounding, and a worst-unaligned local
  page union. Its arithmetic is internally consistent under those assumptions.
- The wrong-token hostile test rejects before output creation or input access.
  The valid-token/missing-review test reserves only an incomplete output.
- The stage has no NumPy import and delays its CuPy import until after its own
  review/source/stream-lock sequence. Numerical fitting is written with CuPy.
- The prose correctly limits A1 to a tied, factorized, row-stochastic,
  state-emitting persistent-regime HMM. It explicitly leaves edge-emitting WFA,
  terminal parity, arbitrary MPS, TTN, and MERA open.

The RunPod command executed only `MathAndModelTests`,
`ArithmeticAndLedgerTests`, and `StrictJsonAndLifecycleTests`: 13 passed, zero
failed in 0.143 seconds. The copied package was deliberately isolated from all
payloads. Its two repository-relative `FrozenPackageTests` were not run from
that relocated copy; exact closure was checked independently instead.

## Blocking defects

### B01 — the advertised holdout is not nested

`run_a1` chooses `chi` and period using all source traces, then passes those
choices into `_crossfit_selected`. Only the seed/model parameters are refit
inside a layer/expert exclusion fold. Held-out layers and experts therefore
participate in hyperparameter selection. The report cannot call this an
untouched whole-layer/whole-expert test. Select every hyperparameter using
training data inside each fold, or freeze the cell before seeing the panel.

### B02 — fold identity becomes a model selector

The crossfit constructs one separately fitted model for each
`(layer_group, expert_group)` pair, aggregates all model packets into global
bytes, and assigns each held-out stream its designated fold model. That makes
the fold/stream identity an effective decoder probability-model selector even
though the design forbids layer, expert, and identity keys in probability
contexts. A cross-validation estimate may use different ephemeral models, but
it is not one identity-free physical packet. Do not label the pooled estimate
as such, or define and charge an explicitly permitted serialized selector under
a revised contract.

### B03 — Q0.16 parameters do not define a bit-exact decoder

The packet serializes integer `pi`, `T`, and emission values, but filtering,
normalization, dense matrix multiplication, reductions, and conditional
frequency rounding are CuPy float64 operations. No independent arithmetic
decoder consumes `winner_model.bin`, reproduces every Q0.16 frequency, and
recovers the selected bits. The operation/reduction order is not a portable
decoder specification. One threshold-crossing float discrepancy desynchronizes
the arithmetic stream. Training may remain CuPy, but final probabilities need
a specified deterministic integer/fixed-point kernel (a CuPy raw kernel is
compatible with the GPU policy) and an independent decode/hash test.

### B04 — the favorable hard-kill bound scores the wrong probabilities

`score_hmm` accumulates NLL from the unrounded float conditional probability.
`exact_hmm_payloads` later rounds each conditional to Q0.16 before arithmetic
coding. Per-symbol rounding can improve the probability of the realized bit,
so `ceil((unrounded_NLL-2)/8)` is not a rigorous lower bound on the rounded
physical payload. An apparent miss can therefore suppress exact coding of a
real survivor. The early-kill bound must use the exact decoder Q0.16 frequency
sequence, or a proven bound that includes worst-case rounding gain.

### B05 — controls skip mandatory baseline replay

The source panel calls `_replay_original` and verifies logical length, payload
bytes, and byte identity. `_controls_after_survival` opens each control panel
but never performs that replay. The controls therefore do not satisfy the
design's claim that every current arithmetic baseline is reproduced before a
candidate score is trusted.

### B06 — final status ignores the holdout result

The A1 status becomes `SURVIVE_EXACT_SOURCE_REQUIRES_HOLDOUT_AND_CONTROLS` from
the directly fitted in-panel packet. The code then computes holdout and controls
but never changes that status if pooled holdout misses, has an invalid packet,
or has a negative minimum fold. The final result can still say `SURVIVE` after
the required validation fails. Final promotion must be explicitly conditional
on every frozen gate.

### B07 — the documented HMM law differs at chunk start

The design writes `A_(c,y)=T diag(E[c,:,y])` and applies it to every symbol.
The implementation emits the first symbol directly from `pi` and applies `T`
only for `time>0`. Both are valid conventions, but they are different laws.
The packet/decoder contract must specify the implemented first-symbol exception
or change the implementation.

### B08 — A0 reset is not a full suffix reset

`_suffix_ids` constructs suffix histories over an entire trace and zeros only
the state exactly at indices divisible by 4096. For the next `depth-1` symbols,
higher lags still include bits from the preceding chunk. This contradicts the
declared reset-every-4096 context. Mask all cross-boundary lags. This defect is
limited to the A0 local control, but the frozen output would currently misstate
that control.

### B09 — CLI path resolution defeats leaf-symlink rejection

The main path calls `.resolve(strict=True)` on review, stream-lock, and control
lock arguments before `HeldRegularFile` uses `O_NOFOLLOW`. Resolution follows a
leaf symlink and hands the target path to `O_NOFOLLOW`; the tested direct-link
rejection does not exercise this main-path behavior. Preserve the absolute
unresolved leaf and open it directly, or use an equivalent descriptor-relative
policy.

### B10 — the frozen search is not operationally feasible

Every dense HMM step is driven by Python time loops with many small CuPy
kernels, dense `chi x chi` operations, backward passes, and `cp.add.at`. The
full grid has 45 `(period,chi,seed)` fits and 12 EM iterations. Even before
constant factors, its transition work scales as

```text
3 periods * 3 seeds * 12 iterations * N * sum(chi^2)
= 589,248 * N
```

for `chi={4,8,16,32,64}`, plus backward, emission, scoring, exact coding, and
potentially many three-seed crossfits. Kernel-launch serialization makes the
implementation worse than the arithmetic count suggests. A practical test
needs a measured synthetic throughput gate and vectorized/fused or
structured/sparse transitions before any payload run.

### B11 — runtime source verification bootstraps from unverified code

`stage0_census.py` dynamically executes `mps_common.py` at module import and
later dynamically executes `verify_source.py`; the latter is asked to verify
the package that supplied the code already executing. A post-review mutation
can therefore run before closure/review rejection. The A469 event was benign,
but demonstrated that members can differ from a still-authentic manifest.
Launch through a small separately reviewed bootstrap that hashes the manifest
and every member before importing or executing any producer module.

## Scope limitations, not additional source defects

1. The experiment models selected SC arithmetic-decision bits and
   decoder-regenerated side contexts from the existing codec. It does not model
   a canonical source-coordinate reconstruction-label stream. A negative result
   closes only this physical decision-stream HMM cell, not the central
   source-coordinate copula/MPS hypothesis.
2. The ladder omits `chi=2`. Although a larger state family can sometimes embed
   a two-state process, the frozen EM search and larger serialized overhead do
   not make this an operational two-state test.
3. A miss cannot be generalized beyond the exact state-emitting factorization,
   `chi<=64`, periods at most four, reset 4096, 12 EM iterations, and three
   starts. In particular it says nothing conclusive about edge-emitting parity,
   a generic nonnegative MPS/WFA, a Born MPS, TTN, or MERA transform.

## Audit access attestation

The auditor did not open, stat, hash, or enumerate any Qwen/model payload,
current finite artifact, extracted decision array, or Gaussian control. The
auditor did not import CuPy, initialize CUDA, or launch a GPU job. Only producer
source, bound metadata hashes, and isolated standard-library source tests were
used.
