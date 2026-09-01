# Independent result audit: nonlinear meta-codebook stage 0

## Verdict

**PASS — KILL confirmed for the frozen latent4 nonlinear decoder cell only.**

The downloaded result is internally consistent and its finite artifacts match
their declared hashes. Both predeclared source seeds miss the favorable
Gaussian-residual target by about a factor of ten in `F`:

The exact nine-file result closure now resides outside the sealed source package
at `research/meta_codebook_whole_expert_stage0_v0_runpod_result_20260901`.
The verifier binds this final sibling path and all member hashes.

| Seed | Source q | Source F | Source s | Gaussian q | Matched s |
|---:|---:|---:|---:|---:|---:|
| 2026090101 | 0.5267904359028147 | 8.078992994248496 | -1.5070877397594313 | 0.5272509448296360 | 0.0006303111048831 |
| 2026090102 | 0.5230531982815714 | 8.021677761279010 | -1.5019520064156389 | 0.5235721704689940 | 0.0007153645276248 |

The charged six-panel first-stage requirement was
`q <= 0.05216397006684782`; the better seed has `q=0.5230531982815714`.
Its `F=8.02167776127901` is 10.0271 times the `F<=0.8` target.

## What this result does and does not kill

The training traces show severe utilization collapse. At update 512 the
source runs use only 1,252 and 1,344 codes in a 2,048-vector minibatch despite
the nominal 32,768-code latent table. The held-out source and matched Gaussian
errors are both near 0.52, with only 0.00063–0.00072 bpw matched advantage.

This is valid negative evidence for exactly the frozen architecture:

- four-dimensional latent table;
- conditional tanh decoder `8 -> 64 -> 64 -> 8`;
- discarded learned encoder and VQ training recipe;
- 512 updates and the two predeclared seeds;
- exact nearest search over the decoder-generated output vectors;
- charged row moments, index bits, decoder and latent table;
- ideal Gaussian residual after the finite first stage.

It is **not** a bound on a direct 32,768-by-8 output-space codebook. The
decoder-generated vectors occupy the image of a shared latent4 nonlinear map;
a direct K-means table gives every eight-dimensional centroid independent
coordinates. Exact nearest assignment removes encoder error, but it cannot
remove the decoder's restricted/collapsed reconstruction family. Therefore the
result does not kill direct output VQ, arbitrary K-means, SoftBinary Coding, or
other nonlinear codebooks.

The result also remains only a six-expert experiment. Its 128-expert ledger is
correct arithmetic but contains no layer-wide generalization evidence.

## Byte and arithmetic findings

- `result.json`: 47,607 bytes, SHA-256
  `9d3f43c8c417e0c9f84849e7a27e6feebbd952a97a74ef29a2307cb933f2a5a0`.
- Canonical result lock:
  `38970e2aafaddd224cd5a103a332b738a99cad555615fe3208eb18ec6a4a6509`.
- Four global binaries are exactly 278,528 bytes. Their `MCBWES0\0` headers,
  version, `K=32768`, latent dimension four and 15-bit index fields are valid.
- Every global FP16 latent/decoder value is finite and every 1,776-byte
  terminal padding region is zero.
- Four moment binaries are exactly 55,296 bytes, byte-identical with SHA-256
  `f734342eafe04669d634f8dd0c520c029875b942d66d7c714e4055b6f2acfe69`;
  all FP16 fields are finite.
- All eight binary hashes equal their result declarations.
- Pooled, expert and matrix `q=SSE/energy`, `F=q*2^(2*prefix)`, `s`, matched
  differences, six-panel ledger, hypothetical-128 ledger and KILL rule are
  independently recomputed by `verify_audit.py`.
- Source receipts are checked for declaration/observation equality only. The
  audit does not open any BF16 source.

Local PowerShell byte/finiteness/hash/oracle QA passed 49 aggregate checks.
The standard-library verifier supplies the full portable replay.

## Replay

Set `INT2_PROJECT_ROOT` to the checked-out `INT2_Q_C` directory (the directory
that directly contains `research/`), then run:

```bash
INT2_PROJECT_ROOT=/workspace/INT2__compression/INT2_Q_C
/usr/bin/python3.12 -B -I \
  "$INT2_PROJECT_ROOT/research/meta_codebook_whole_expert_stage0_v0_result_audit_20260901/verify_audit.py" \
  --root "$INT2_PROJECT_ROOT"
```

The verifier uses only source JSON and the downloaded result/side artifacts.
It imports no CuPy and opens no BF16 payload.
