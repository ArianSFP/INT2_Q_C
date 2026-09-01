# FUSEED-PMG1-v2 red-team decision

## Verdict

**The narrowed architecture is scientifically defensible under explicit
conditions; it is not yet executable.** Control movement does not inherently
invalidate the claim because the controls cease to be exhaustive matched
searches. Candidate multiplicity is handled by a one-descriptor validation
firewall. The controls are only a restrictive, fixed-descriptor diagnostic and
capture subtraction after selection. This can reject a survivor but can never
select a replacement.

## The invalid claim that is expressly rejected

It would be invalid to search 2^32 source seeds, evaluate only the source winner
under 32 controls, and then describe those controls as if each had received the
same 2^32 search. They did not. PMG1-v2 therefore claims no exchangeability,
randomization p-value, familywise p-value, or matched-search winner's-curse
correction. Any document making such a claim blocks the package.

## Why held-out validation is enough for the narrower positive claim

The seed is a deterministic function of selection data. Once it and the entire
selection state are committed, its result on a genuinely untouched validation
panel is a single pre-specified evaluation. The cardinality of the training
search does not create 2^32 validation comparisons. A strict no-retry rule is
essential: changing the seed, ABI, coordinates, K, threshold, control rule, or
arithmetic after validation would silently turn the validation panel back into
selection data.

## Remaining blockers

1. An independent party must prove the ABI narrowing was based only on the
   already-bound public-source family and source-free feasibility evidence.
2. The validation panel must be demonstrably untouched for every decision in
   this protocol and unavailable during selection.
3. The source-free retention suite must pass all 256 prospectively mapped cells
   with the locked simultaneous lower bound.
4. A complete-shape calibration must pass the unchanged 900-second limit and
   close compiler identity, dual-reference parity, generator-state parity,
   exact-plan and journal/global-merge gaps from the v1 audit.
5. An independent source-only audit must authenticate the frozen successor
   before any payload run can be authorized.

If either chronology or validation independence cannot be proved, the proper
verdict is **BLOCK**, not “exploratory evidence.”
