# Independent CCQ stage-0 result audit — 2026-09-01

Verdict: **PASS — the sealed result's `KILL` decision is confirmed within its
stated narrow scope.** This does not mean the compression target passed. It
means the source repair, byte bindings, canonical packets, arithmetic, and
negative conclusion are internally consistent.

The immutable evidence audited is:

- `research/ccq_raw_mse_stage0_v0`
- `research/ccq_raw_mse_stage0_v0_runpod_result_20260901`

Neither evidence directory was modified. This audit did not open model or BF16
payloads, use a GPU, or remeasure SSE against tensors. It independently replays
all arithmetic from the locked matrix SSE/energy rows and validates the result
packets byte-for-byte.

## Exact byte bindings

The repaired source package has nine regular, non-link files. Its manifest is
`e8beee384b1d4de37010b5b97ed4f412f147d826547d8dc6b2934a2bd795c78a`.
The repaired verifier is
`f8c2e3c1396295109461bdafa8552bede3e4619abc87217ada9bedbee428f570`.
The runner stayed
`7288be471925fe9596c76b6f39a814e3a2589fc3857b10f7387ca9cd2271f474`.

The result directory has exactly three regular, non-link files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `source_prefix.bin` | 7,518,592 | `125c5bb318b93ab49dd5ba0d42ee0f2068648547dc151d29b72010f81de3ab1b` |
| `gaussian_prefix.bin` | 7,518,592 | `cffc586d515c70ee98320d394730032e97d689d490a6b45ce6dcff7ee8286ec3` |
| `result.json` | 49,169 | `f48a7462aa18fcf973c8b1bdb76544d851439a2036cd79fc58d29016dbcc93c4` |

Removing `result_lock_sha256` and hashing the strict canonical JSON gives the
declared result lock
`d458ba0f642f02b515aee2e5726dfd00b541ef782710089014a515b87387f63a`.
The pretty `result.json` bytes also reproduce the producer's sorted, finite
JSON serialization exactly.

## Packet audit

Both packets have an exact 4,096-byte global header followed by six exact
1,252,416-byte expert frames. Each expert contains:

- 64 header bytes;
- 1,179,648 index bytes;
- 36,864 packed uint4 local-scale bytes;
- 28,672 little-endian float32 code-scale/zero-point bytes;
- 7,168 canonical little-endian FP16 super-scale bytes.

The audit independently checks the global and expert magic/version/identity,
canonical compact metadata, all zero padding, every field boundary, finite
float32 parameters, positive code scales, finite positive FP16 super-scales,
exact end offset, absence of trailing bytes, and byte-identical parse/rebuild.
Each packet contains 43,008 float32 and 21,504 FP16 parameter values. Both
canonical roundtrips pass.

## Rate and read replay

The six-expert fixed prefix is 7,518,592 bytes over 28,311,552 values, or
2.1245298032407407 bpw. Its containing-oracle requirement is
`q <= 0.0420722358191473`.

| Final R | Physical bytes | Ideal residual bytes | Cold expert bytes (4 KiB) | Read amplification |
|---:|---:|---:|---:|---:|
| 2.15 | 7,608,730 | 90,138 | 1,273,856 | 1.0045219110153731 |
| 2.30 | 8,139,572 | 620,980 | 1,363,968 | 1.0054346837892705 |
| 2.50 | 8,847,360 | 1,328,768 | 1,478,656 | 1.0027777777777778 |

All index, local-scale, per-output float32, FP16, and framing bytes are charged.
Every cold read is below 2x. The 128-expert numbers also recompute exactly, but
are only an amortization projection and provide no 128-expert evidence.

For an ideal Gaussian residual using all remaining bits, the exact identity is
`F = q * 2^(2 * prefix_bpw)` at every listed final rate. The audit re-derives
this identity separately at 2.15, 2.30, and 2.50 bpw. The oracle is deliberately
favorable and is not a finite residual stream.

## Metric and kill replay

| Panel | Fit q | Held-out q | Held-out F | Held-out s |
|---|---:|---:|---:|---:|
| Source | 0.13872340614366618 | 0.13829419641954802 | 2.629652429483857 | -0.6974360629913025 |
| Matched Gaussian | 0.1370967296373507 | 0.13742826460002708 | 2.6131868092920847 | -0.6929051312546927 |

Every one of the 36 matrix `q = SSE / energy` rows, both fit/holdout pooled
aggregates, all four held-out expert aggregates, `F`, `s`, and the matched gap
recompute. Source minus Gaussian `s` is `-0.00453093173660979`: the natural
source is slightly worse than its moment/RMS-matched Gaussian control. The
largest recorded row-mean and centered-RMS matching errors are respectively
`1.0170566611122922e-9` and `1.8231747084263006e-9`.

Using complete held-out energy as the denominator, accumulated held-out source
SSE gives `F_lower_bound = 0.8422425704563797` after matrix ordinal 4. Remaining
SSE is nonnegative, so no later matrix or ideal residual can recover `F <= 0.8`.
The final pooled source `F = 2.629652429483857`, and both held-out experts also
fail. `KILL` is therefore forced.

## Claim boundary

The evidence kills only the frozen, paper-derived CCQ Code-Cluster cell on this
authenticated six-expert panel. It is not an official CCQ encoder reproduction,
a finite residual codec, fresh validation, a model-wide or 128-expert result,
or target achievement. The source-only receipt's `gpu_execution: not run` is a
snapshot of source-package sealing before the separate coordinated result; it
is not treated as a claim that no later run occurred.

## Replay

From a checkout root containing `research/`:

```bash
/usr/bin/python3.12 -B -I \
  research/ccq_raw_mse_stage0_v0_independent_result_audit_20260901/verify_audit.py \
  --root "$PWD"

/usr/bin/python3.12 -B -I \
  research/ccq_raw_mse_stage0_v0_independent_result_audit_20260901/test_audit.py
```

The verifier and all 15 hostile regression tests use only the Python standard
library. Tests cover strict JSON duplicates/non-finite overflow, manifest
closure, a real symlink (or explicit platform skip), trailing packet bytes,
header padding, wrong label, and non-finite/non-positive serialized parameters.
