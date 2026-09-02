# TACTIC-CAGE coarse-programmed graph/Krylov oracle v0

Date: 2026-09-02

Status: frozen source-only Qwen pilot gate. No Qwen/model payload, completed
TACTIC N18-v6 result, BF16 source, matched control, CUDA device, or network was
opened while this package was built or tested.

## Question this package answers

This package tests the strongest falsifiable part of TACTIC-CAGE before a
finite graph codec is built:

> Does a graph determined only by the already-paid N18-v6 coarse symbols
> contain enough of the exact original-domain coarse residual to permit the
> 2.5-bpw target in an extremely favourable continuous oracle?

It consumes one literal, terminally completed
`tactic_actual_coarse_n18_v6` result and the exact authenticated BF16 triplet
that produced it. It independently reopens and hashes the complete result,
replays the authenticated v6 decoder, checks literal re-encoding, and scores
FP64 residuals in canonical Gate/Up/DownT source coordinates.

This is **not a finite codec**. It transmits no coefficients, supports, graph
selector, or decoder. Its best-support projection gives 384 exact real
amplitudes and their support away for free in every 4,096-weight block. Its
waterfill score assumes ideal independent-Gaussian coding of graph spectral
coefficients at exactly 384 bits per block. Both are optimistic containment
oracles, not achieved packets.

## Frozen decoder-visible graph

Each full N18 record has 262,144 canonical I32 inverse-RHT symbols. It is split
into 64 contiguous 4,096-coordinate blocks. For each block, three fixed path
graphs are constructed without reading the BF16 source or residual:

1. `coarse_signed_path_dct`: stable order by `(coarse_symbol, coordinate)`;
2. `coarse_magnitude_path_dct`: stable order by
   `(2*abs(coarse_symbol)+sign, coordinate)`;
3. `coarse_context_path_dct`: stable order by a frozen integer combination of
   the current and public XOR-neighbour coarse symbols, then coordinate.

The public control is `public_coordinate_path_dct`, which retains native
coordinate order. Each path uses its exact orthonormal DCT-II graph Fourier
basis. The path Laplacian eigenspaces are the spectral envelope approached by
Lanczos/Chebyshev Krylov filters, so the result is labelled a
graph-spectral/Krylov containment oracle. No eigensystem, edge, permutation,
or fitted transform is transmitted.

CuPy performs sorting, graph ordering, DCTs, energy reductions, top-k
containment, reverse waterfilling, and controls. Exact source-domain totals are
accumulated and checked in FP64. The v6 runtime lock pins CuPy/CUDA/device.

For every integer subspace rank `k=1,...,384`, the result emits the fixed-rank
capture curve, the isotropic fixed-subspace baseline `k/4096`, their excess,
the source-leaking top-k curve, and Qwen-minus-each-control excess curves. Raw
continuous capture never promotes the architecture: promotion uses only the
Qwen-minus-matched-control excess of the ideal rate-shaped score.

## Frozen budgets and gates

The physical coarse rate is exactly `307/128 = 2.3984375` bpw. The remaining
2.5-bpw budget is exactly 384 bits per 4,096 weights:

```text
R_fine = 13/128 bpw
B_block = 4096 * 13/128 = 416 bits
```

TACTIC's frozen layout reserves 32 of those bits for non-coefficient fields,
so this oracle uses the requested 384-bit DH384 correction budget. It never
claims that 384 real coefficients can be carried by 384 bits.

Let `D0` be the actual pooled N18-v6 relative MSE and `D*=0.025`. The exact
required residual-energy capture is

```text
c_required = max(0, 1 - D*/D0).
```

The raw N18-v6 pilot currently awaiting its independent result audit reports
`D0=0.036975150060595235`, so the provisional direct threshold is
`c_required=0.323870222053737` (32.3870% of coarse SSE), not the earlier
19.1% finite-baseline figure. Its raw `COARSE.bin` and `RESULT.json` hashes are
recorded in `design_lock.json` strictly as pending evidence. The runner will
not trust them without the terminal closure and all caller-supplied hashes.
Ideal Gaussian use of the remaining `13/128` bpw preserves the coarse
`F=1.02781086823352`; the graph branch must expose source-specific structure,
not merely spend the extra rate.

The gates are applied in this order:

1. Any source/result/hash/re-encode/FP64/orthonormality failure aborts.
2. If even the best graph's free-support, free-amplitude top-384 projection
   has final pooled relative MSE above `0.025`, the family is
   `HARD_KILL_CONTINUOUS_GRAPH_ENVELOPE_MISSES_TARGET`.
3. Only a source survivor opens controls. The source-free package does not
   pre-open, pre-hash, or enumerate a control.
4. The primary rate-shaped gate uses ideal 384-bit reverse waterfilling. If its
   best graph misses `0.025`, status is
   `HARD_KILL_IDEAL_384BIT_WATERFILL_MISSES_TARGET`.
5. Define graph advantage in rate-equivalent units as
   `0.5*log2(D_public/D_graph)`. The graph must beat the public coordinate
   basis by at least `0.03 bpw`, and its Qwen advantage must exceed the larger
   of the exactly block-moment-matched Gaussian and within-block permutation
   controls by at least `0.03 bpw`. Otherwise it is
   `HARD_KILL_COARSE_GRAPH_NOT_SOURCE_SPECIFIC_0P03_BPW`.
6. A surviving ideal result is `ELIGIBLE_FOR_FINITE_GRAPH_LIFTING_BUILD`.
   It is marked composite-gap eligible only if the control-subtracted
   advantage is at least `0.11356063 bpw`.

These thresholds can kill this frozen graph/Krylov family. They cannot kill
all nonlinear TACTIC-CAGE trees or prove a finite codec.

`COARSE.bin` is opened exactly once. The literal bytes, decoded I32 coarse
state, F32 reconstruction, and FP64 residual are buffered; graph work may scan
those buffers but cannot refetch the compressed expert file. The result keeps
three disjoint ledgers: external compressed storage, host parse/buffer/scratch,
and accelerator HBM. Only the first is a cold-storage read. HBM kernel traffic
is not measured, so the package has no below-2x inference-HBM authority.

## Controls

Controls are conditional-residual controls, not end-to-end Gaussian PTQ runs:

* a frozen odd-affine permutation independently shuffles each 4,096-value
  residual block while leaving the coarse graph unchanged;
* a frozen counter-generated Gaussian block is centered and scaled to match
  that block's exact FP64 residual mean and centered energy, again leaving the
  coarse graph unchanged.

They test whether the graph is aligned with non-Gaussian/Qwen residual
structure rather than merely benefiting from an orthogonal basis and oracle
allocation. A future promotion still requires the separately built full
matched-Gaussian PTQ producer.

## Source-only verification

Run before any payload launch:

```bash
python -I -B research/tactic_cage_graph_krylov_oracle_v0/verify_source.py \
  --package research/tactic_cage_graph_krylov_oracle_v0
python -I -B research/tactic_cage_graph_krylov_oracle_v0/test_source_only.py
```

The tests use only synthetic arrays and temporary directories. They assert
that CUDA/CuPy is not imported, exercise exact budgets and decision
boundaries, validate waterfilling and controls, and hostile-test source/result
closure and atomic publication helpers.

## Explicit pilot launch

Every mutable external object is pinned by a caller-supplied SHA-256. The
output directory must not exist and must remain outside both immutable source
packages and the v6 result.

```bash
python -I -B research/tactic_cage_graph_krylov_oracle_v0/run_oracle.py \
  --authorization RUN_TACTIC_CAGE_GRAPH_KRYLOV_ORACLE_V0_QWEN_PILOT \
  --package-manifest-sha256 <this-SOURCE_MANIFEST.json-sha256> \
  --v6-package research/tactic_actual_coarse_n18_v6 \
  --v6-package-manifest-sha256 <v6-SOURCE_MANIFEST.json-sha256> \
  --v6-predecessor-lock-sha256 <v6-PREDECESSOR_LOCK.json-sha256> \
  --v6-runtime-lock-sha256 <v6-RUNTIME_LOCK.json-sha256> \
  --v6-result-dir <literal-completed-v6-result> \
  --v6-complete-sha256 <literal-COMPLETE.json-sha256> \
  --input-manifest <exact-v6-input-manifest> \
  --input-manifest-sha256 <exact-input-manifest-sha256> \
  --output-dir <absent-output-directory>
```

The output is an atomic, no-replace directory containing `RESULT.json`,
`PROVENANCE.json`, and terminal `COMPLETE.json`. It always sets
`positive_claim_authority=false`, `finite_codec_executed=false`, and
`inference_read_claim_authority=false`.

Three bounded decoder-legal follow-ons are frozen in
`SECONDARY_SCREENS.md`: coarse-signature collaborative patch groups,
coarse-seriated Hankel/displacement rank, and non-dyadic Ramanujan phase
energy. `secondary_hooks.py` records their buffer-only routing and common
control-subtracted gate. They are not executed by this pilot; bispectral
Volterra remains a conditional later gate.
