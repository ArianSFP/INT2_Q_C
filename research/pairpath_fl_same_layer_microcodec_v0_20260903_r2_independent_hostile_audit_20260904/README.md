# Independent hostile audit: PAIRPATH-P2 r2

## Verdict

**BLOCK — no payload capability and no hard-kill authority.**

The exact target closure was independently authenticated at manifest SHA-256
`21983efff5ac5c0593a655cae4136d35ca24400fd807f9fe4be458a34b18e622`
and source root
`7ffb0b9c92861c7171a3b89f47d6fa03caac963322d772fb8c0b020ce501cf96`.
Its own verifier and ten source-only tests pass. The execution boundary is also
real: no Qwen/model payload, GPU/CuPy, network, RunPod, or execution authority
was used by this audit.

Three material defects prevent progression to a local-RTX-3060 payload run.

1. The sealed design and README claim one global Up/Down rate-distortion
   multiplier. The finite `_make_plan()` path instead calls
   `choose_pair_labels()` separately for each role, and that function derives
   the bit weight from that role's energy. On the hostile unequal-energy
   fixture, the correct global weight is `0.12503048310980563`, while the
   observed Up and Down weights are respectively `0.000989431291029312` and
   `0.24907153492858197`.

2. The optimistic joint solver is alternating and has no upper-envelope or
   global-optimality certificate. A deterministic legal-level counterexample
   shows that it returns joint objective `11.976051614873764` even though the
   label assignment found by the independent solver is a valid joint-model
   candidate with objective `11.901158026223808`. The gap is
   `0.07489358864995665`. Consequently, a low reported gain can be an optimizer
   failure and cannot safely authorize the advertised hard kill.

3. The literal decoder never validates or replays the transmitted tree
   descriptor. It accepts `bits=0`, `packed=1`, duplicate pair `[0,0]`, and
   materialized leaves `[9,9]`, then reconstructs and scores the packet. This
   violates the fail-closed packet/selector contract even though ordinary
   source, packet, decoded-label, and decoded-scale hashes otherwise bind real
   bytes correctly.

## Checks that passed

- Exact target closure, manifest/root, and all ten target tests.
- Sixteen constant label-pair starts plus nearest/equal starts.
- Role-conditioned nearest-label mutual information, independently recomputed.
- Equal-rate and equal-MSE formulas, hull monotonicity/convexity, and units.
- IID kill and perfectly aligned survive synthetic mechanism tests.
- Literal source/packet/label/scale binding and source-tamper rejection.
- Independent recomputation of both padded and conservative read ledgers; the
  tested packet's maximum amplification is `1.2286464191976003x`.
- Affine-value and Gaussian controls rerun the complete fitting, selection,
  packet, decode, and score pipeline.
- Exhaustive tree descriptor replay for 2, 4, 6, and 8 experts.
- All execution flags disabled and `run_gate.main()` fail closed.

## Required repair

Use a single Up/Down energy normalization in the finite joint search; make the
joint oracle dominance-safe by seeding/evaluating every independently optimized
label solution under the joint law (and do not call it a hard upper envelope
without a real certificate); and validate/replay the exact two-expert tree
descriptor plus canonical stream directory during packet parsing. Reseal as a
new source revision and repeat this audit. Do not mutate the sealed r2 closure.

`AUDIT_REPORT.json` contains the exact machine-readable measurements.
