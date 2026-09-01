# Qwen result: charged sparse-tail peeling is globally killed

## Decision

The frozen exact-lossless-tail architecture does **not** reach the required

```text
F = MSE * 2^(2R_actual) <= 0.8.
```

The best charged construction is support-pattern XKLT at exactly `2.5 bpw`:

| Quantity | Result | Required |
|---|---:|---:|
| Ideal source-relative MSE | **`0.030619458569072736`** | `<= 0.025` |
| `F` | **`0.9798226742103275`** | `<= 0.8` |
| `s=-0.5 log2(F)` | **`0.01470370863865911 bpw`** | `>= 0.16096404744368115 bpw` |
| Fraction of required `s` | `9.134778152123511%` | `100%` |
| MSE below same-rate Gaussian | `2.0177325789672462%` | `>= 20%` |
| Excess over target MSE | `22.47783427629093%` | `<= 0%` |
| Worst cold expert read | **`1.0392171223958333x`** | `<2x` |

The winning configuration peels only `93 / 28,311,552` weights
(`0.00032848782009548613%`) carrying `0.022546455341546085%` of panel energy.
Its complete side ledger is `42,973 bits` (`0.001517860977737992 bpw`), and
the remaining ideal residual payload is `70,735,907 bits`. Larger exact tail
planes cost more support/value bits than their removed energy can repay.

This is a hard kill for the frozen grid, not merely a coordinate-search miss.
The Lagrange-dual certificate enumerates all `20^3=8,000`
tail-choice triples inside each of six experts for both raw and
support-conditioned XKLT residual geometries. Its weakest lower bound over all
three rates and both geometries is

```text
F >= 0.979822670284832 > 0.8.
```

The best constructed row is only `3.925495484224939e-09` above that certified
lower bound. Weak duality therefore excludes every one of the frozen
`20^18` panel choice vectors, including configurations not visited by
coordinate descent.

## Rate sweep and certificates

| Actual physical rate | Best exhibited support-XKLT `F` | Dual lower `F` | Ideal MSE | Target MSE | Worst cold read |
|---:|---:|---:|---:|---:|---:|
| `2.149999830457899` | `0.9798242589551213` | `0.9798242487578499` | `0.0497415493375334` | `0.0406126293632105` | `1.045603280127338x` |
| `2.2999999434859664` | `0.979823578951942` | `0.9798235722689215` | `0.040402658277502654` | `0.03298770035374648` | `1.042628659421977x` |
| `2.5` | **`0.9798226742103275`** | **`0.979822670284832`** | **`0.030619458569072736`** | **`0.025`** | **`1.0392171223958333x`** |

For comparison, the globally certified raw-bulk lower bounds are
`0.9964256887873799`, `0.9964247885955159`, and `0.9964235802429079` at the
same three rates. Support-conditioned role innovation is useful, but sparse
exact tail peeling contributes nowhere near the missing `0.14626033880502204`
bpw.

## Why the leaky envelope passes

If the exact tail values and XKLT bases are revealed to the decoder for free
while only the mask is charged, the 2.5-bpw diagnostic reaches
`F=0.6960072100696455`. It does so by peeling the maximum tested `12.5%` of
all weights, which carry `51.33602998286245%` of source energy. The mask alone
costs `0.5450560958297165 bpw`.

That row is intentionally illegal. The sharp contrast with the fully charged
optimum shows that the bottleneck is transmitting millions of exact BF16 tail
values, not residual transform quality or MoE locality. It does not motivate
a finite polar-lattice run under this exact-value architecture.

## Scope and execution

The CPU-only pilot opened the pinned Qwen3-30B-A3B panel:

- `18` authenticated Gate/Up/Down matrices;
- `6` experts and `28,311,552` BF16 weights;
- panel FP64 source energy `16192.894508855932`;
- all `18/18` source SHA-256 identities matched; and
- no Gaussian controls and no CUDA work were run after the charged early-kill
  and complete-grid certificate succeeded.

Exact command:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  /workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/tail_peeling_composite/tail_peeling_composite.py \
  --source-lock /workspace/INT2__compression/blind_protocol_v2/unblinded/source_hashes.lock.json \
  --source-root /workspace/INT2__compression/blind_protocol_v2/unblinded \
  --output /workspace/INT2__compression/INT2_Q_C/research/tail_peeling_composite/pilot_qwen_numpy_dual/result.json \
  --backend numpy \
  --control-replicates 0 \
  --maximum-coordinate-passes 5
```

The independent verifier rechecked the result seal, exact mask lengths,
value-code bounds, candidate choices, complete side and payload closure,
component RD identities, all integer allocations, the dual/primal inequality,
both-geometry/rate coverage, read ledgers, and all eighteen source files.

Evidence:

```text
result JSON SHA-256   f2c3c1006d274d9025c1a61e3da9549bf22cd61bc333457c8198a34c538e2cf2
result internal lock f7d38fb4f67b77c5af577d8123286e4b26ded8b7f387f05a91f8b37faa85f547
oracle SHA-256        f8c211520ba78a1e96dedbd6b7225a25a21928254f9578ba38835be729825f4f
protocol SHA-256      30145dea08b24c97f2025e2c04c59d414347c50e089d5bc375d5066b5a052f64
verifier SHA-256      e696469078a9eda01291a75ecae9ec470f60816df93026adbe787366ea47db9a
```

The retained artifacts are
`pilot_qwen_numpy_dual/result.json` and
`pilot_qwen_numpy_dual/verification_receipt.json`.

## Claim boundary

The certificate kills the exact frozen family: stable top-absolute supports,
lossless BF16 tail values under the declared literal/Huffman codes, raw or
support-pattern XKLT residuals, and ideal RHT/polar-lattice RD. It is not a
converse for lossy tail reconstruction, structured/semantic masks, learned
value predictors, activation-weighted loss, or arbitrary nonlinear codecs.
