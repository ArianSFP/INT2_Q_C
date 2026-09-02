# TACTIC-CAGE review — universal SwiGLU-MoE PTQ

Date: 2026-09-02

## Verdict

TACTIC-CAGE is mathematically legitimate, but the proposal contains both real
open branches and mechanisms that cannot create information.  The immediate
experiment order is:

1. run the authenticated long-range UWFA source census on the current Qwen
   artifact;
2. if a conditional law survives, measure a cross-fitted posterior centroid
   for the complete literal message;
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
  detector calibration, not Qwen evidence.
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
2. **Posterior gate:** cross-fit the smallest exact-cell centroid first.  Keep
   only a joint `G_joint` improvement; never add separately fitted rate and
   MSE gains.
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

The codec may use only the literal packet, public shape, Gate/Up/Down role and
fixed algorithmic constants.  Model/checkpoint/layer/expert identity, a public
base checkpoint, Qwen-fitted parameters, router activations or uncharged graph
edges are forbidden.  Qwen is an evaluation panel, not part of the decoder
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
