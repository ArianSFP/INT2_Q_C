# Universal SwiGLU label-copula census — sealed source-only stage 0

This package freezes a narrow experiment for the central source-model question:

> After removing ordinary public marginals, do canonical SwiGLU-MoE weight
> labels contain enough held-out, nonlocal dependence to save at least
> `0.1528899669629145` physical bits per source weight?

That is the standalone saving needed to move the unchanged audited finite
reconstruction from `s=0.008074080480766676 bpw` to `F<=0.8`.  The package does
not add gains from separately fitted oracles.

At seal time it is **source-only**.  No checkpoint/model tensor, current codec
payload, decoded decision stream, Gaussian control, network resource, NumPy or
CuPy array, CUDA context, or GPU job was opened.  The launch surface has no
payload authority.  An independent source review must bind this exact source
manifest before a separate payload adapter may be written or run.

## What v0 tests

The primary view, A, is deliberately simple and universal:

1. An authenticated adapter identifies semantic Gate, Up, and Down tensors.
2. Gate and Up have semantic shape `[d_ff,d_model]`; Down has semantic shape
   `[d_model,d_ff]`.
3. Every expert is traversed in micro-neuron order
   `Gate[j,k], Up[j,k], Down[k,j]`, for arbitrary positive `d_ff,d_model`.
4. Each public run of 2048 scalar weights is divided by its RMS.
5. A fixed standard-Gaussian symmetric Lloyd-4 threshold
   `0.981598821873` assigns ordered labels `0,1,2,3`.
6. Labels use ordered Gray codes `00,01,11,10`; the two decisions are emitted
   MSB then LSB.

RMS calculation and Lloyd labeling are encoder-side source diagnostics.  They
are not part of the probability decoder and they do not establish a usable
weight reconstruction.  In particular, this raw-label census omits scales and
reconstruction levels.  It is **not yet a complete 2.15–2.5 bpw codec**, and a
large raw-label saving cannot be copied into the current codec without one
nested finite re-encode.

View B—current STRATA transformed indices or bitplanes in their natural block
and coordinate order—is deferred.  It requires a distinct, independently
authenticated extractor, byte-exact replay of the current arithmetic payload,
and Gaussian controls passed through the complete frozen STRATA encoder.  v0
does not resolve or open those artifacts and cannot claim a result for B.

## The exact-integer unifilar bank

Every candidate has one public reset length in

```text
32, 64, 128, 256, 512, 1024, 2048, 4096 binary decisions
```

and one state size in

```text
chi = 2, 4, 8, 16, 32, 64.
```

The five topology families are:

- `suffix`: the last `log2(chi)` decisions;
- `parity_sketch`: `log2(chi)` deterministic long-range XOR sketches, with
  final public check positions protected from overwriting the sketch;
- `modular`: a public-coordinate-weighted modular prefix sum;
- `rolling`: a truncated polynomial rolling state;
- `regime`: the last symbol plus a saturating run-age state.

This gives exactly 240 nonlocal candidate cells.  Every update is O(1),
integer, deterministic, and unifilar.  The decoder-visible context contains
only semantic role, Gray plane, phase modulo eight, clipped distance to the
public reset boundary, and causal state.  Checkpoint/model identity, layer,
expert, absolute tensor site, source weight, future decision, and the
source/control flag are forbidden probability keys.

For context `c` and state `z`, training counts are Jeffreys-smoothed and rounded
by exact integer arithmetic to `f1[c,z] in [1,65535]`.  The binary row is

```text
f(1|c,z) = f1[c,z]
f(0|c,z) = 65536 - f1[c,z].
```

Thus every row sums to 65536 exactly.  The physical model is a 256-byte header
plus the complete dense context-by-state uint16 table.  The topology selector,
state size, reset length, and table are all decoder-visible and charged.  No
float defines a decoded probability or successor.

The package includes a real 32-bit binary arithmetic encoder and decoder.
Literal termination bits, final-byte padding, model bytes, directory bytes,
per-expert frame headers, 64-byte frame alignment, final container-page
padding, and all cold-read pages are included.  Every scored outer-test frame
must independently reconstruct all labels from the serialized uint16 model and
public role/plane schedule.

## Nested evaluation—not full-panel selection

Layer and expert labels are permitted only as partition metadata.

- Outer test layers are selected by a frozen hash order.  Every expert in
  those layers is untouched until the final score.
- Among non-test layers, frozen expert slots form the inner validation fold.
  Those expert slots are absent from training.
- All 240 cells are fit on training and ranked on actual validation container
  bytes, charging a complete model and all finite framing.
- A separate `chi=1` factorized bank independently selects its reset on the
  same validation fold.
- Only the winning topologies are refit on training plus validation and coded
  on the untouched whole-layer test.

The paired point source statistic is

```text
8 * (factorized_test_container_bytes - nonlocal_test_container_bytes)
---------------------------------------------------------------------
                         test source weights
```

not an ideal cross entropy and not a source-minus-control difference.  The
implementation also performs 4096 deterministic paired whole-test-layer
bootstrap replicates, allocating the exact shared-byte delta by layer source
weight.  Its lower 95% bound—not merely the point estimate—must exceed
`0.1528899669629145 bpw`.

The factorized comparison is deliberately strong: it uses the identical
public role/plane/phase/boundary context, independently chooses its reset on
validation, uses the same quantized probabilities and arithmetic coder, and
receives its own literal model/framing/page ledger.  Therefore a gain is
attributable to causal state rather than to public position or marginal
nonstationarity.

## Matched-Gaussian controls cannot create a pass

Eight seeds are frozen.  For raw view A, every control must independently:

1. sample Gaussian values matched to each source 2048-weight block's mean and
   variance;
2. rerun canonical orientation, RMS normalization, Lloyd labeling, and Gray
   serialization;
3. rerun the complete nested train/validation/test model search;
4. refit its independently selected source model;
5. arithmetic-code and charge its own physical packets and reads.

Controls may not even be generated until the source has already cleared the
absolute `0.1528899669629145 bpw` gate.  Control subtraction may measure
source-specific excess, but **no control-created pass** is possible.

A Qwen survivor would still be evidence on one evaluation family, not proof of
universality.  The frozen algorithm must transfer to at least one independently
sourced, architecturally different SwiGLU-MoE before a universal claim.

## Exact storage and cold-read accounting

One diagnostic container charges:

- a 4096-byte container header;
- the selected model rounded up to complete 4096-byte pages;
- 64 directory bytes per expert, with the directory page-padded;
- a 256-byte header in every expert frame;
- the exact arithmetic payload bytes, including termination and byte padding;
- every expert frame rounded to 64-byte storage alignment;
- final container padding to a complete page.

For one routed expert, the read ledger includes the container header, every
selected-model page, one addressed directory page, and the exact set of pages
intersecting that expert's frame.  The report includes maximum cold-read
amplification.  This diagnostic ledger is not substituted for the final MoE
codec's `<2x` ledger; a survivor must later be embedded into one unchanged
finite reconstruction and measured again.

## Why this closes a real gap, but not every tensor network

A bounded suffix model can miss parity and other high-order relations at
arbitrary separation.  The source-free fixture constructs matched-marginal
blocks whose final six decisions are deterministic XOR checks of a 26-decision
random body.  A six-bit parity sketch retains the relevant information across
the gap; a short suffix has collisions.  This proves that the tested nonlocal
bank is meaningfully broader than the prior local-context screen.

A miss still closes only the frozen 240-cell bank and its fixed public context.
It does not close arbitrary edge-emitting weighted automata, learned
nonnegative MPS, Born MPS, TTN/MERA transforms, site-specific models, nonlinear
flows, or a different reset/context family.  Conversely, a fixture success is
only an implementation check and says nothing about real model weights.

## Fail-closed lifecycle

The wrong authorization token rejects before output creation, input
resolution, source verification, CuPy import, or CUDA.  A correct token first
reserves an absent output directory and marks it incomplete, verifies the
sealed source, authenticates a separate independent source-review receipt,
and reads only a metadata lock that explicitly has no payload authority.  CuPy
is imported only after those gates; future GPU optimization is CuPy-only.  v0
then emits a preflight receipt with zero payload opens and zero kernels.
`COMPLETE.json` is created exclusively last.  An incomplete directory is never
resumed or accepted.

The launch gate is intentionally not a payload runner:

```bash
python3 -B -I research/label_copula_census_stage0_v0/stage0_census.py \
  --authorization OPEN_AUTHENTICATED_LABEL_COPULA_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V0 \
  --review-receipt /absolute/path/independent_source_review.json \
  --input-lock /absolute/path/source_only_input_metadata.json \
  --output /absolute/path/new-absent-output-directory
```

## Package members

- `label_copula_common.py`: exact extraction reference, topology bank,
  quantized models, arithmetic codec, split, selection, and ledgers;
- `run_source_free_fixture.py`: nonlocal/matched-control implementation probe;
- `stage0_census.py`: fail-closed CuPy/payload lifecycle preflight;
- `test_source_only.py`: arithmetic, geometry, split, control, and hostile
  lifecycle tests;
- `design_lock.json`: machine-readable frozen protocol and claim boundary;
- `verify_source.py`: independent-friendly package closure/static verifier;
- `SOURCE_MANIFEST.json`: exact file hashes and byte counts.

## Mathematical lineage

The experiment is a finite-state operational test, not an appeal to an ideal
entropy theorem.  MPS Born machines motivate the possibility of high-order
label laws, while weighted finite automata provide a small exactly causal
subclass.  The relevant primary references include:

- Han et al., *Unsupervised Generative Modeling Using Matrix Product States*,
  https://arxiv.org/abs/1709.01662
- Srinivasan et al., *Quantum Tensor Networks, Stochastic Processes, and
  Weighted Automata*, https://arxiv.org/abs/2010.10653

Those works motivate the census.  They do not provide evidence that Qwen—or
any other SwiGLU-MoE—contains the required `0.1528899669629145 bpw` structure.
