# Independent hostile audit: TETRAPATH-4 source oracle v0

## Verdict

**FAIL AS A HARD-KILL OR PROMOTION ORACLE; PASS AS A SOURCE-ONLY MECHANISM
FIXTURE.**

The package is genuinely source-only and does not claim a Qwen result.  Its
tuple geometry, probability normalization, Chow-Liu enumeration, pairwise
maximum-entropy fit on the audited control, XOR synergy fixture, balanced-IID
fixture, and ideal common/private fiber ledger all pass.

The reported `HARD_KILL_MEMORYLESS_FOURWAY_BELOW_0P045_BPW` status is not
scientifically authorized, however.  The search is a finite collection of
alternating local fits on a sparse multiplier grid.  It is not a dominant
upper envelope.  A negative result can therefore be a false negative.

## Blocking findings

### B1 — the optimizer is not global, with an exact two-coordinate counterexample

`optimize_family` alternates between fitting a probability model and choosing
coordinate-wise labels under that frozen model.  Thirty-four symmetric starts
and twelve alternations do not exhaust the label assignments.

The audit exhaustively enumerated all `256^2 = 65,536` full-model assignments
for a deterministic two-coordinate distortion field.  At multiplier `1/2`:

```text
reported alternating-search objective  0.8234051108809574
exact exhaustive objective              0.8210407366447210
gap                                      0.0023643742362364
exact tuple ids                          [74, 74]
```

The complete deterministic costs and source energy are retained in
`AUDIT_EVIDENCE.json`.  The target test called
`test_assignment_equals_exhaustive_tiny_global_search` checks only assignment
under one *fixed* probability vector.  It does not test the alternating
fit-and-assign optimization that produces the frontier.

This is fatal specifically to a hard-kill claim: missing a better challenger
point can understate the four-way gain.

### B2 — smoothing breaks containment of the lower-order families

The full model uses a Jeffreys-like `+1/2` count for all 256 tuples, while the
independent model smooths four separate four-symbol marginals.  These are not
the same universal-code penalty, and the fitted full family no longer
operationally contains the independent family.

For the exact singleton assignment `[0,0,0,0]`, the target reports:

```text
independent fitted rate   1.0000000000000000 bpw
full fitted rate          1.6065661886755245 bpw
full minus independent    0.6065661886755245 bpw
```

At Qwen-sized samples this particular smoothing effect may be numerically
small, but the claimed dominance is a mathematical property, not a
large-sample heuristic.  It must be enforced, not assumed.

### B3 — the sparse multiplier grid is not frontier-complete

Only sixteen fixed multipliers from `0` through `4` are searched.  Convexifying
the points found on this grid cannot recover an omitted rate-distortion point.
Using the same grid for all families provides procedural symmetry, not a bound
on the gain between grid points.  This independently prevents a `<0.045 bpw`
result from being a hard kill.

### B4 — `G4` is not isolated four-way connected information

The status gate uses full versus the envelope of independent, 2+2, and
Chow-Liu families.  It does not use the stronger all-pairwise maximum-entropy
surrogate already computed by the package.  Moreover, pairwise maximum entropy
removes pair interactions but leaves third- and fourth-order effects combined.
Calling the remainder `G4` or pure four-way synergy is therefore too strong.

Use full versus pairwise maximum entropy for a general `order_ge_3` diagnostic.
If pure fourth-order interaction is the claim, also fit a maximum-entropy law
matching every three-way marginal.

### B5 — the gated survivor need not be routably local

The status is decided by the unrestricted full 256-state family.  A survivor
there does not imply that any of the three equal-fiber common/private families
survives.  The unrestricted four-expert-coordinate stream has no demonstrated
sub-2x routed-read representation.

For the MoE objective, report two distinct decisions:

1. scientific higher-order structure detected;
2. read-compatible fiber projection detected.

Only the second can advance directly toward a finite routed-expert codec.

## Non-blocking limitations

- `aligned_up_down_values` validates four equal FP64 shapes but does not
  transpose raw Down matrices or authenticate that a caller already did so.
  The orientation contract is explicit, but it is not mechanically proven.
- All fitted tables, model selection, and time sharing are free.  This is
  acceptable for a favourable mechanism screen, never for a finite result.
- The fiber amplification is an entropy-level logical ledger.  Page rounding,
  headers, common-page placement, and concurrent expert routing are not yet
  represented.

## Passing evidence

- All 12 target unit tests pass.
- The target source verifier passes and prints
  `PASS_UNSEALED_SOURCE_ONLY_TETRAPATH4_NO_PAYLOAD_AUTHORITY`.
- No network, model-download, CuPy, Torch, or payload import exists in the
  audited code.
- All fitted distributions normalize within `1e-11` in the independent audit.
- The unsmoothed all-pairwise IPF fit matches every audited pair margin within
  `1.03e-15` absolute error.
- The XOR fixture has zero pairwise MI, `0.25 bpw` full-joint advantage,
  `0.75 bpw` total fiber rate, and `4/3` ideal maximum routed-read amplification.
- The balanced IID fixture has exactly zero measured higher-order gain.
- The payload gate raises unconditionally; there is no Qwen, GPU, network, or
  deployment authority.

## Required repair before payload execution

1. Rename the present result an `INCONCLUSIVE_HEURISTIC_SCREEN`; remove hard-kill
   authority.
2. Use an operational probability treatment that preserves model containment,
   or explicitly inject every lower-order fitted law as a legal full-family
   candidate and verify pointwise dominance.
3. Add exact branch-and-bound/exhaustive certificates on tractable blocks and
   an auditable optimality-gap bound on production blocks.  More random starts
   alone do not create a hard-kill proof.
4. Adaptively refine the multiplier grid until a declared frontier-error bound
   is met.
5. Gate `order_ge_3` against pairwise maximum entropy; add all-three-way maximum
   entropy before using a pure-four-way name.
6. Require a separate equal-fiber survival result for the `<2x` MoE route.
7. Bind raw Up/Down shapes and perform the transpose inside a future one-use
   capability.

The package remains useful: it correctly demonstrates why pairwise-null data
can contain large higher-order synergy and gives a promising constructive
common/private mapping.  It is not yet able to close the branch on a negative
Qwen run.
