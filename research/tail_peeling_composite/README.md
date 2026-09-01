# Charged sparse-tail peeling composite

## Result

**Globally killed.** On all 18 pinned Qwen matrices, the best charged ideal-RD
row has `F=0.9798226742103275`, `s=0.01470370863865911 bpw`, MSE
`0.030619458569072736` at `2.5 bpw`, and worst cold expert reads
`1.0392171223958333x`. A complete-grid Lagrange-dual certificate proves
`F>=0.979822670284832` across the frozen tail grid, so no finite CUDA codec is
warranted. No matched controls or CUDA work were launched after that gate.

See [RESULT.md](RESULT.md) and the independently checked
[`pilot_qwen_numpy_dual`](pilot_qwen_numpy_dual/) evidence. Exact artifact
sizes and SHA-256 identities are sealed in
[`ARTIFACT_HASHES.json`](ARTIFACT_HASHES.json).

This package tests the one important tail architecture that the earlier
nonparametric, STRATA, scale-field and composite studies do not actually
cover: losslessly peel individual large BF16 weights, charge both their
locations and values, and only then transform and allocate the robust bulk.
See [COVERAGE_AUDIT.md](COVERAGE_AUDIT.md) for the precise subsumption audit.

## Frozen architecture

For every matrix, the search chooses one of twenty nested stable top-absolute
supports, from zero through 12.5% of the matrix. Ties are resolved by the
canonical flat source ordinal. The physical side ledger contains:

- `ceil(log2 C(N,k))` bits for a combinatorial-number-system mask;
- an exactly counted lossless BF16 value stream, selected from four canonical
  literal/Huffman modes;
- a 128-bit descriptor per matrix, a 64-bit residual directory per live joint
  waterfill component, expert headers, the literal route table and a 4-KiB
  global header; and
- 16 bits per support-conditioned XKLT Givens angle.

Known residual supports are partitioned into the seven nonempty Gate/Up/Down
patterns. A source-fitted orthogonal KLT is applied inside each pattern, then
a procedural RHT feeds an ideal Gaussian polar-lattice test channel. All
components across all six experts share one water level. The final continuous
allocation is rounded to exact integer payload bits by assigning each
remaining bit to the largest exact marginal distortion reduction.

The ideal residual channel is an optimistic lower bound on achievable MSE. It
has no finite-length, lattice-shaping, RHT-padding, angle-quantization or
arithmetic-coder loss. Coordinate descent supplies an exhibited construction,
not a global claim. A separate certificate enumerates every `20^3` tail triple
inside each expert and evaluates the Lagrange dual of the six-way discrete
choice plus continuous payload problem. The family is killed only if that
weak-duality lower bound also has `F>0.8` for every rate and both residual
geometries. A charged pass is only permission to build and decode a real
container.

## Decision rule

At each actual byte-capacity rate `R` in `2.15, 2.30, 2.50`:

```text
F = ideal_relative_MSE * 2^(2R)
pass iff F <= 0.8
```

Every tail byte and residual bit lies in an expert-local frame. The report
reconstructs cold byte and 4-KiB-page reads from the selected per-component
integer allocations and requires the maximum cold byte amplification below
`2x`.

The preregistered interpretation layer can run four independently searched
moment/energy-matched Gaussian controls through the same BF16 value alphabet,
tail grid, entropy ledger, support XKLT and waterfill. They were deliberately
not run: the absolute complete-grid Qwen lower bound already hard-killed the
architecture, so controls could only interpret a failure, not rescue it.

## Local tests

```bash
python research/tail_peeling_composite/tail_peeling_composite.py --self-test
python -m unittest research.tail_peeling_composite.test_tail_peeling_composite
```

The tests use synthetic sources and do not open the pinned panel or require a
GPU.

## Optional full matched-control command (not executed)

The implementation remains CuPy-ready, but the certified kill does not warrant
this additional GPU/control run. If the protocol is extended and a new tail
family first survives its absolute gate, do not reuse an existing output path;
the oracle refuses to overwrite it.

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/tail_peeling_composite/tail_peeling_composite.py \
  --source-lock /workspace/INT2__compression/blind_protocol_v2/unblinded/source_hashes.lock.json \
  --source-root /workspace/INT2__compression/blind_protocol_v2/unblinded \
  --output /workspace/INT2__compression/tail_peeling_composite_run_1/result.json \
  --backend cupy \
  --control-replicates 4 \
  --maximum-coordinate-passes 5

/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/tail_peeling_composite/verify_tail_peeling_result.py \
  --result /workspace/INT2__compression/tail_peeling_composite_run_1/result.json \
  --source-root /workspace/INT2__compression/blind_protocol_v2/unblinded
```

The executed CPU-only early-kill command and its source scope are recorded in
`RESULT.md`. `protocol_lock.json` freezes the grid, rates, controls, physical
ledger and promotion rule.

## Claim boundary

This is an ideal-RD architecture oracle, not an emitted codec or achieved
weight reconstruction. It rejects or promotes this exact-lossless-tail
family. It does not rule out lossy tail quantization, learned semantic masks,
activation-weighted objectives or arbitrary nonlinear compressors.
