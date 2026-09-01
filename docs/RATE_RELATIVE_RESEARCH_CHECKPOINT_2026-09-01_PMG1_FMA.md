# Rate-relative Qwen checkpoint — PMG1 explicit FMA and fresh validation

## Status

This checkpoint preserves a source-free compute breakthrough and the audit
findings that constrain its next experiment. It does **not** claim that the
final distortion target has been reached and it authorizes no model or
validation-payload access.

The acceptance rule remains

```text
2.15 <= R_actual <= 2.5 bpw
F = MSE * 2^(2*R_actual) <= 0.8
s = -0.5*log2(F) >= 0.16096404744368115 bpw
maximum cold compressed read per routed expert < 2x
```

The independently decoded finite baseline is unchanged: exact `R=2.5`, MSE
`0.030902167403153148`, `F=0.9888693569009007`, and worst cold page read
`1.1694444444444445x`. At that rate the final MSE requirement is `0.025`.

## FUSEED-PMG1 stage-0 result

PMG1 tests one prospectively fixed Megatron-compatible Philox initialization
ABI over the complete `2^32` base-seed space. Its stage-0 source objective is
binary64 throughout, reloads decoded FP16 affines, ranks finite capture by
descending binary64 value and then ascending u32 seed, retains exact Top-8192,
and stores packed 12-byte `(seed_u32, capture_f64)` records. The candidate has
no per-weight seed table: a final descriptor and eighteen FP16 affine pairs
would occupy 80 bytes over the six-expert target scope.

Two exact-shape source-free calibrations reached different decisions:

| Arithmetic | Projected complete stage 0 | Frozen gate | Decision |
|---|---:|---:|---|
| contraction-disabled binary64 v1 | `696.8922319519334 s` | `<650 s` | kill |
| explicit-FMA binary64 v2 | **`520.8358833260136 s`** | `<650 s` | stage-0 survivor |

The v2 source prospectively spells ten `__fma_rn` sites and nineteen explicit
rounding-intrinsic sites, with `--fmad=true`, `--ftz=false`, precise division,
and precise square root. This is not a relabeling of v1: the independent audit
found 119 changed binary64 capture encodings among the common Top-K seeds and
a changed ordering. Three complete-shard replays were deterministic. Their
median end-to-end shard time was `2.0215128749841824 s`; the projection includes
finite checks, exact Top-K, packed journal fsync, cold excess, all 256 shards,
and a global-merge shape probe.

The independent standard-library audit performed `148,185` checks and
authenticated the explicit-FMA derivation, the ABI1 plan, both compiled-kernel
hash receipts, CUDA header/runtime bindings, all packed records, canonical
Top-K, projection arithmetic, and three separate direct/sequential/Torch BF16
parity receipt families. Its exact verdict is
`BLOCK_PAYLOAD_AUTHORIZATION_STAGE0_V2_SURVIVOR_AUTHENTICATED`.

## Why v2 cannot run on Qwen

The old draft called experts `[24,56,88,120]` untouched validation. The audit
proved that claim false: two earlier frozen result receipts already contain
target-derived fit and score moments for all eight corresponding Up/Down
matrices. The issue is not cosmetic and cannot be repaired retrospectively.
The v2 design remains blocked even though its stage-0 timing passed.

Other unclosed v2 prerequisites are a reproducible retention stress, exact
stage1/stage2/validation timing, serialized cubin and full toolchain closure,
crash-safe descriptor-relative journals, two byte-identical legal merge trees,
and a durable one-descriptor/no-retry commit before validation visibility.

## Fresh v3 validation precommit

Before fetching any replacement tensor byte, a distinct PMG1-v3 precommit now
chooses validation identities using only a public hash rule. Rank

```text
SHA256("FUSEED-PMG1-v3|fresh-validation|layer=15|expert=E")
```

for every `E in [0,127]` not divisible by eight and take the four smallest.
The fixed rank order is **`[67,95,69,34]`**. Gate, Up, and Down are all bound,
so Gate cannot be deferred to an adaptively chosen second panel. The twelve
exact tensor names, shapes, shards, relative offsets, and inclusive HTTP
ranges are sealed against the pinned Qwen revision. Only the already-public
safetensors index/header metadata was inspected; the twelve data ranges were
not fetched or materialized by the precommit turn.

The replacement ranges remain forbidden until a new v3 design passes source
audit and retention/full-pipeline calibration, and one winning descriptor,
every selection state, applicable threshold, fixed controls, and a no-retry
sentinel have been durably committed.

## Parallel release-gate findings

Two other candidates were stopped before accelerator or model use:

- Grouped layout-overlay v5 passed its producer tests but its independent
  source audit found an import-and-call bypass, a create-new TOCTOU, and
  pathname execution after authenticated descriptors were closed. A distinct
  v6 descriptor/capability repair is required.
- Lossy-tail v7 passed 28 producer adversarial tests, but a fresh audit found
  five release blockers: mutable parent-cmdline provenance, authorization-
  selected audit status, pathname output TOCTOU, incomplete stable-order GPU
  memory evidence, and pool release with live GPU-bearing locals. Runtime
  calibration was correctly withheld.

These are release-conformance failures, not positive or negative distortion
results. Neither branch contributes a score to PMG1 or the finite baseline.

## Exact artifact ledger

```text
PMG1 explicit-FMA script              0e2f354415d2d8cfebfceda58b6ade77eddc2b4e025488baba329beba09d0a87
PMG1 explicit-FMA result              82e29cbfc8ec1ac23761c37712a3fda3d2745b04c9a71ae296ce864796ddc75e
PMG1 explicit-FMA package manifest    c7d163c0271999a5c8c70adb4bd717055585e156ccb751528c0acffade577a24
PMG1 independent audit manifest       d69d144824ceffb2e7249e7a86be04296e98fd6f8a74429d7562ef5fea8b257d
PMG1 independent audit receipt        84fffb44df6da89af1e229aa2f117cd3e87352ca13663b58a9bdfdbbc0b3fe94
PMG1 fresh-validation precommit       2585838ba9abb28e83fbc26045dbae1aa243b680c04164cbc576731738bc5852
PMG1 fresh-validation manifest        522887a6c3954c55a049cc6163290ea26bf412ff395f39adc12e0dbacd1a8090
Grouped-v5 producer manifest          e9ef19853d1350ac4085a18a65b19a6805620ef7a091447ce57404715f88805f
Grouped-v5 audit receipt               59e6b5ed1fff09801939ae5c737214546f20a3708a2803a88e1b9e0c0432fe7f
Lossy-tail-v7 producer manifest        3d5bc5ed95071cc45406d0d2906b54f40d32adad0dffc6323b8fa80ca491ed63
Lossy-tail-v7 independent audit        120b616c726253a82850e93b720e48c56c2aa7af59f1c2b7ec288bec215e4621
Lossy-tail-v7 audit receipt            b82146a04188b74a3213fd54db2ee1bb34c7d132dd685b8169b7f8ce36a78dff
```

The explicit-FMA package-manifest hash above is the SHA-256 of
`artifact_sha256.txt` after its seven payload hashes were frozen.

## Verification

From the repository root on PowerShell 7:

```powershell
research/fuseed_pmg1_binary64_fma_calibration_v2/verify_result.ps1
research/fuseed_pmg1_v3_fresh_validation_precommit/verify_precommit.ps1
research/lossy_tail_peeling_oracle_v7_fresh_source_audit/verify_audit.ps1
```

The PMG1 independent audit itself is a Linux standard-library replay:

```bash
/usr/bin/python3 -B -I \
  research/fuseed_pmg1_binary64_fma_calibration_v2_independent_source_audit/audit.py
```

The next promotable step is a separately sealed PMG1-v3 design. If its
retention suite or total calibrated projection fails, the exhaustive run is
killed without reading selection or validation payload. If both pass, only
the existing auxiliary selection panel may open first; the fresh twelve-range
validation panel still waits for the durable one-shot commit.
