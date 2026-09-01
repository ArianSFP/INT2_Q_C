# BiSCo run_1 independent replay

This note records the state-backed replay of the frozen auxiliary-only BiSCo
`d=16, h=64, 18+18` experiment.  The replay ran on the supplied RunPod with
CuPy 14.2.0 on an NVIDIA GeForce RTX 5090.  It did not open the pinned Qwen
panel.

## Verdict

The replay confirms `HARD_KILL_D16_SHALLOW_BEFORE_PINNED` at update 512.

| Statistic | Frozen result | Independent FP64 replay | Absolute difference |
|---|---:|---:|---:|
| `D_Qwen` | 0.11020813815423404 | 0.11020813758494276 | 5.6929128e-10 |
| `D_Gaussian` | 0.10961316220157559 | 0.10961316325045290 | 1.0488773e-09 |
| `s_match` | -0.003904857950836688 | -0.003904847322141395 | 1.0628695e-08 |

The original FP32 reduction was also emulated independently and reproduced all
16 published matrix SSE values exactly.  The maximum relative FP64-SSE
difference was `3.3301501363830925e-08`; the maximum independent source-energy
difference was `1.2299459866810625e-15`.

This result is far below the matched-source promotion requirement: its
`s_match` is slightly negative, not the required positive margin.  The kill is
only for the frozen shallow training recipe.  It is not a converse for deeper
learned codebooks or unrelated nonlinear PTQ architectures.

## What was independently reconstructed

The audit implementation in [`../independent_replay.py`](../independent_replay.py)
does not import or call the training oracle.  It:

- defines its own exact FP32 state shapes, offsets, order, and little-endian
  parser;
- requires the exact frozen result, state, decoder, launch, and 32 auxiliary
  source hashes;
- proves both FP16 decoder files are byte-identical to the decoder fields from
  their respective FP32 states after IEEE-binary16 rounding;
- requires exactly history updates `[256, 512]`, recomputes every derived
  evaluation field, requires exact final/history equality, reconstructs the
  complete early-kill object, and derives the top-level decision;
- independently recomputes training-role Qwen moments from the frozen BF16
  sources, regenerates the eight held-out filename-seeded Gaussian controls,
  and checks charged FP16 normalization records;
- evaluates the 512-update states through a separately written FP32 encoder,
  decoder, and fixed-order greedy bit-flip path; and
- computes source-domain SSE by converting the normalized FP32 residuals to
  FP64, multiplying by the exact stored-FP16 RMS, and reducing in FP64.

Each per-matrix receipt row also binds the final 36-bit code stream and
normalized FP32 reconstruction with SHA-256.

## Tolerances

The replay receipt records both limits and observed errors.  The main SSE
comparison uses

```text
rtol = gamma_128 = 128 * 2^-24 / (1 - 128 * 2^-24)
atol = 1e-10
```

This conservatively covers FP32 squaring plus a hierarchical FP32 reduction;
the replay accumulation itself is FP64.  The observed maximum relative error
(`3.33e-08`) is over two orders of magnitude below that limit.  The separate
same-device FP32-emulation check uses `rtol=5e-7` and observed exact equality.
Independent blocked FP64 energy/moment reductions use 64 binary64 ulps plus
`1e-12` absolute tolerance.

## Sealed bindings

| Object | SHA-256 |
|---|---|
| `bisco_raw_mse_result.json` | `5904e3887e69cf47ee4a882aeaacceb27823504c1e23eeff6adb4b3360874d92` |
| `independent_replay.py` | `0a3c38fab4cbac640b1731e66b72e94fefe370b015a9cbe99d19a585114d0bd1` |
| `independent_replay_receipt.json` | `f75fc33b9b67cb3b711e2f54a95994757ed062364d980ad9849458e69feb76e7` |
| canonical unsigned receipt | `4ca302c0232640efa34b072baec50c74ec295046f53c57e65973b7212345c04b` |
| 32-source filename/hash root | `d7e892bc3dfbee8668fa474c1e96ef0db9f7fab3d918dc90a818fb4d7f5f5ca6` |

The receipt seal is SHA-256 over sorted-key, compact ASCII JSON with the
`receipt_seal` field omitted.  Verification additionally rebinds the seal to
the local script, result, states, and decoder files and recomputes receipt
aggregation rather than trusting the stored `verified` flag.

## Verification

From the repository root:

```bash
python research/bisco_raw_mse_oracle/independent_replay.py \
  --verify-receipt research/bisco_raw_mse_oracle/run_1/independent_replay_receipt.json

cd research/bisco_raw_mse_oracle
python -m unittest -v test_bisco_raw_mse_oracle.py test_independent_replay.py
```

For a fresh source-backed GPU replay, follow the command in the package
[`README.md`](../README.md) and write to a new receipt filename.
