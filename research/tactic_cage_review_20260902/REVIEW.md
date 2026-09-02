# TACTIC-CAGE review — universal SwiGLU-MoE PTQ

Date: 2026-09-02

## Verdict

TACTIC-CAGE is mathematically legitimate, but the proposal contains both real
open branches and mechanisms that cannot create information.  The immediate
experiment order is:

1. run the authenticated long-range UWFA source census on the current Qwen
   artifact;
2. retain the fitted non-local trace and run a bounded cross-fitted posterior
   centroid for the complete literal message; a failure of the standalone rate
   gate lowers its prior but does not mathematically rule out within-cell MMSE;
3. build and independently decode a real `307/128`-bpw coarse packet before
   testing DH384 or a coarse-programmed graph;
4. promote a graph only after a decoder-legal continuous oracle survives;
5. implement an adaptive 384-bit tree only after the graph survives;
6. attempt syndrome coding or bits-back only after measuring the conditional
   source law that would make it useful.

This order is deliberately different from building the full CAGE backend
first.  The UWFA experiment is already sealed and directly tests the cheapest
remaining information source.  The current N18 work does not yet contain an
eligible lower-rate coarse codec.

## What remains genuinely open

- A model-charged, whole-component-held-out WFA/CTW/TTN law on actual Qwen SC
  decisions.  The `0.16754150390625` bits/symbol result is a source-free
  detector calibration, not Qwen evidence.  It must not be compared directly
  with the `0.1528899669629145` bits/weight ideal target: the units differ.
  This artifact has `4.472635975590459` selected SC symbols per weight, so the
  stricter page-aligned same-reconstruction threshold is
  `0.03441710571244585` saved bits per selected symbol after every model and
  framing byte.  The fixture's magnitude is therefore ample as a detector
  calibration, but says nothing about Qwen's actual saving.
- A non-local posterior centroid conditioned on the complete decoded message.
  Existing local LUT, affine and small deterministic decoder corrections do
  not contain a persistent/non-unifilar latent posterior over exact SC cells.
- A graph/lifting transform derived only from an actual lower-rate coarse
  codeword, public role and shape.
- A nonlinear prefix-adaptive 384-bit refinement codebook conditioned on that
  coarse word.
- A coarse-residual syndrome code, but only if measured conditional entropy or
  conditional rate-distortion makes its 384-bit message useful.

## Already adverse or partially contained

- RAVEL's source-leaking held-out correction captured only
  `0.0005730920838821207` of baseline SSE, and its finite table was worse than
  identity.  This closes another local centroid/LUT, not a non-local posterior.
- The generic cyclostationary/Hankel rank-four Qwen oracle was hard-killed at
  best `F = 0.9379899307967997`.  Revisit only with a genuinely
  coarse-derived seriation.
- MALT64 and other fixed local decoded tangents are adverse for another fixed
  local linear frame.  They do not contain a decision-dependent nonlinear
  tree, but they lower its prior.
- The aligned-role syndrome opportunity was far below the standalone target.
  A coarse-residual conditional syndrome is a different experiment.
- Arithmetic-coder overhead is about `0.0000433405` bpw and cannot supply the
  missing gain.

## Prior-art boundary

The abstract ingredients are not individually novel.  Conditional entropy
models and coarse-to-fine quantized hierarchies are established in learned
compression (Mentzer et al., arXiv:1801.04260; Duan et al.,
arXiv:2208.13056).  Recent LLM work also contains adaptive linear transforms
(WUSH, arXiv:2512.00956), Haar-wavelet weight quantization (HBLLM,
arXiv:2512.00862), and sigma-delta weight representations (SDQ-LLM,
arXiv:2510.03275).  Posterior-mean reconstruction under MSE and syndrome/coset
coding are classical principles.

The defensible research-novelty hypothesis is narrower: an independently
decodable physical coarse MoE weight packet programs an expert-local graph and
refinement codebook, the encoder jointly searches the ordinary coarse and fine
messages, and the same packet is posterior-decoded under a literal one-pass
`<2x` routed-read ledger.  The literature search did not identify this exact
combination in weight PTQ, but absence from a bounded search is not a legal or
patentability conclusion.  More importantly, novelty does not imply gain; the
Qwen conditional oracle still decides whether CAGE is worth implementing.

## Mathematical accounting

Let `C` be the transmitted coarse code and let the graph or transform program
be `G = Phi(C, shape, role)`.  Then `H(G | C, shape, role) = 0`: no graph
descriptor bytes are needed.  This does **not** imply free residual
information.  A hash-selected transform class cannot carry extra information
without restricting the available coarse codewords, whose cost must appear as
coarse distortion in a joint coarse/fine search.

For a complete literal message `M`, the posterior centroid is valid because

```text
E ||X - Y(M)||^2
  = E ||X - E[X|M]||^2
  + E ||E[X|M] - Y(M)||^2.
```

The opportunity must be learned on disjoint experts/layers or a separate
family and charged physically.  A source-fitted centroid table is not a
universal constant.

A 384-bit adaptive decision tree has exactly `2^384` leaves per coarse word.
A 384-bit syndrome also selects at most `2^384` decoder reconstructions per
coarse word.  Adaptivity can improve geometry and search; a coset can improve
conditional coding; neither increases transmitted information.

Rate and MSE must be evaluated in one reconstruction.  A useful diagnostic is

```text
G_joint = (R_baseline - R_candidate)
          - 0.5 * log2(D_candidate / D_baseline).
```

The ideal target is `G_joint >= 0.1528899669629145` bpw.  For the current
page-aligned same-reconstruction artifact, the literal saving threshold is
stricter: 544,768 bytes, or `0.153935185185185` bpw.

## Physical and bandwidth constraints

The current 2.5-bpw packet has no metadata slack.  Every graph table, model,
latent, offset, termination state and alignment page must replace payload
bytes.  A small model can cost a complete 4 KiB page.

One exception is decoder-fixed redundancy: the proposed selector table, empty
QC region and seed fixtures are deterministic from the algorithm version.  A
lean deployment format can omit their 20,480 bytes, reaching
`R=2.49421296296296`.  The unconditional same-MSE `F` multiplier is
`0.99200955785219`; applying it to the audited STRATA baseline gives a
favourable planning transfer `F=0.980967853512842`, not an executed CAGE
result.  This modest byte budget may instead fund a charged posterior model,
but it is not close to the target.

The frozen TACTIC planning layout reads six global pages and 359 expert pages
against a 360-page owner share, giving `73/72` for one pass.  Reading the
expert frame twice gives

```text
(6 + 2*359) / 360 = 2.011111...
```

and fails.  A coarse-derived graph must therefore be generated and consumed
from one compressed-frame scan, or the frame must remain resident.  The
physical receipt must report unique pages, repeated requested bytes and the
coalesced interval union; a unique-page union alone does not prove operational
traffic below 2x.

## Missing lower-rate coarse object

`tactic_actual_coarse_n18_v3` proves useful byte-count identities but is not a
finite N18 codec.  Its independent audit found no packet magic/header grammar,
logical EOF, scale/seed/source fields, payload digest, canonical decoder or
re-encoder, and no concrete placement for the former global metadata.  Its
1x/2.5-bpw figures are therefore layout arithmetic, not a compression result.

A successor must emit one versioned packet language with exact universal tail
rules, independent decode/re-encode, repeated-read instrumentation, and a
separate eligibility boundary for shapes that can satisfy `R in [2.15,2.5]`
and `F <= 0.8`.

## Decisive gates

1. **UWFA Qwen gate:** hard-kill the exact recoder unless its literal,
   model-charged, disjoint-component saving reaches the actual page-aligned
   byte threshold and every component is positive.  Generate matched controls
   only for a source survivor.
2. **Posterior gate:** cross-fit the smallest exact-cell/non-local-state head
   first.  Keep only a joint `G_joint` improvement; never add separately fitted
   rate and MSE gains.  A negative label-rate result alone is not a converse
   for a continuous within-cell centroid, so permit one bounded oracle before
   killing the posterior branch.
3. **Coarse graph oracle:** derive every graph operation from the independently
   decoded coarse word.  Charge any allocation or selected component not
   derivable from it.  Score exact inverse reconstruction in source FP64.
4. **Adaptive tree gate:** compare fixed frame, nonadaptive graph, and
   prefix-adaptive graph at the same literal 384 bits.  A beam-search miss is
   not a family converse unless accompanied by a valid upper bound.
5. **Syndrome/bits-back gate:** do not build until a held-out conditional model
   demonstrates enough margin after model, framing, random-access reservoir
   and cold-read costs.

## Universality boundary

The decoder may use the literal packet--including source-fitted parameters
that are serialized and fully charged--plus public shape, Gate/Up/Down role
and fixed algorithmic constants.  Universality does not require a source-free
encoder.  It forbids uncharged Qwen-fitted constants, identity-indexed lookup
tables, a public base checkpoint, router activations, source references or
uncharged graph edges.  Qwen is an evaluation panel, not part of the decoder
definition.  Portability still requires a disjoint SwiGLU-MoE family after a
Qwen survivor.

## Evidence paths

- `research/unifilar_wfa_entropy_census_stage0_v8/`
- `research/nonlocal_wfa_global_state_synthetic_v0_synthetic_runpod_20260901/`
- `research/ravel_decoded_residual_lut_stage0_v1_runpod_result_20260901/`
- `research/cyclo_fri4_normal_stage0_v0_qwen_run_20260901_5fb09e8/`
- `research/tactic_conditional_dyadic_coset_v2/`
- `research/tactic_actual_coarse_n18_v3_independent_audit_20260902/`
- `research/universal_causal_noise_shaping_syndrome_mdl_nogo_v0/`
- `research/tactic_cage_review_20260902/COMPOSITE_GRAMMAR_REVIEW.md`
- `research/tactic_cage_review_20260902/INFORMATION_ACCOUNTING_CORRECTIONS.md`
- `research/tactic_cage_review_20260902/INFORMATION_THEORY_REDTEAM.md`
