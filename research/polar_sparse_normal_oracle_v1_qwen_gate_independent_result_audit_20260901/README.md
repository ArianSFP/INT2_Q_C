# Independent result audit: PSNO-v1 fixed spatial groups

## Verdict

`KILL_QWEN_SPECIFIC_GROUPED_SPATIAL_SIGNAL_ROUTE`.

The source-leaky fixed-group oracle has a numerically feasible ideal-Gaussian
allocation: the best measured family is the 32-by-32 upper-triangular tile
prefix at `F=0.7723583974071416`, or `s=0.18632881917746885 bpw`.  That is not a
finite codec.  Every selected coefficient is reconstructed as its exact
continuous value while being charged only one bit.

More importantly, eight spatially permuted controls preserve every absolute
orthonormal normal coefficient and reproduce the same search.  For the winning
family their mean is `F=0.7723995661350902`,
`s=0.18629037051627295 bpw`.  Qwen's excess is only
`0.00003844866119590007 bpw`; the combined control-MC and delete-one-expert
three-standard-error interval is
`[-0.00002422821679954594, 0.00010112553919134608] bpw`.  All eight tested
families are null-reproduced.

The most favourable source-specific upper endpoint is only 0.063% of the
standalone `0.16096404744368115 bpw` requirement, and 0.089% of the published
composite incremental `0.11356063456788208 bpw` gap.  It is short by factors of
about 1,592 and 1,123 respectively.  The apparent absolute pass is therefore a
generic consequence of the impossible exact-value/one-bit channel and the
idealized component metric, not evidence for exploitable Qwen tile locality.

This audit kills the tested grouped spatial-signal route and does **not** claim
to kill every nonlinear value codec.

## Feasible primal replay

The producer emitted Lagrangian lower bounds.  The independent replay fixes
each dual-selected group count, charges the exact enumerative support and
one-bit value ledger, then reverse-waterfills all remaining model and normal
components to use exactly `R=2.5`.  Seven of eight impossible-channel prefixes
are feasible below `F=0.8`; offset segments of length 8 miss at
`F=0.8003039993690865`.

The winning B=32 allocation uses 2,693 groups, 2,749,200 exact-continuous value
symbols, and 5,317 support bits across the panel.  Its ledger is:

- base side: `0.0011701230649594908 bpw`;
- support: `0.00018780319779007523 bpw`;
- nominal one-bit values: `0.09710523817274305 bpw`;
- total side: `0.09846316443549262 bpw`;
- exactly used Gaussian payload: `2.4015368355645075 bpw`.

## Producer launch-contract limitation

The downloaded producer result and its 88-check receipt are internally sealed,
bind the exact 18 sources, and reproduce all prior binary64 normal hashes.
However, executed runner SHA-256
`e5540d0e9beabb984af15ab569aceac8a29cc0be91286e595d51bfafa3704f08`
does not preflight `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, or
`MKL_NUM_THREADS`, and its own launch example omits them.  A first preflight
without the original contract changed record-0 normal energy by 16 ULP and
changed its normal SHA.  Re-running under the parent producer's documented
`4/4/4` environment restored every tested scalar and the exact normal SHA.

The successful result is scientifically bound by its exact normal hashes, so
this limitation does not change the numerical verdict.  It does mean the
executed source package is not self-containedly reproducible from its own
launcher.  The independent control replay fails closed unless all three thread
variables equal the string `4` before NumPy or CuPy is imported.

## Finite-value replacement decision

No finite-value implementation is warranted from this branch.  The source-
specific signal ceiling is around `1e-4 bpw`, far below either applicable gap,
before paying quantization error, headers, kernels, or codebook reads.

If future independent evidence reopens the route, the strict first gate should
be a leave-one-expert-out 32-by-32 shape codebook or per-tile low-rank family:

1. train no template or factor basis on the target expert;
2. charge exact group support, rank/mode headers, indices, scales, and every
   codebook byte;
3. grant continuous exact scales/factors at their declared finite ledger for
   the containing early gate;
4. repeat the identical eight spatial controls and require positive corrected
   source-specific `s`, as well as absolute `F<=0.8`;
5. only then quantize and measure a finite reconstruction.

A concrete cold-read ceiling remains below 2x: three role-specific codebooks of
64 FP16 32-by-32 shapes occupy 393,216 bytes.  Against a three-matrix expert
frame of 1,474,560 bytes at 2.5 bpw, a single-pass fused decoder would read
`(1,474,560+393,216)/1,474,560 = 1.2666666667x`.  An all-role shared codebook
would be 131,072 bytes and 1.0888888889x.  These are proposed upper ledgers,
not permission to implement after this kill.

## Artifacts

- `control_replay.json`: sealed CuPy Qwen/control replay.
- `replay_controls.py`: authenticated replay producer, including mandatory
  4/4/4 launch preflight.
- `audit_receipt.json`: compact machine-readable verdict and arithmetic.
- `verify_audit.py`: standard-library independent seal/arithmetic verifier.
- `AUDIT_MANIFEST.json`: immutable file hashes.

No fresh validation data was accessed.  The control replay used the same
authenticated 18 matrices only.
