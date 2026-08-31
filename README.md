# INT2_Q_C: auditable sub-2.15-bpw PTQ on Qwen weights

## STRATA-XKLT-SC v2 second blind-panel confirmation — pass

The frozen **STRATA-XKLT-SC v2** PTQ codec passed its precommitted gate on one
deterministic 18-matrix panel from `Qwen/Qwen3-30B-A3B`. The independent
original-source-domain score was **0.04985939119332436** at an exact complete-
artifact rate of **2.149999830457899 bpw**. This is
**1.7854225272960011% below** the strict Gaussian reference
`2^-4.3 = 0.050765774772264724` (`0.07824047385148768 dB`).

The 7,608,729-byte container charges its source-derived KLT metadata, literal
route, labels, profiles, scales, lengths, padding, and zero reservoir tail.
The independent auditor decoded and canonically re-encoded all 14 blocks,
verified 18/18 matrices and 108/108 nested source blocks, and passed its
execution, lineage, source-derived metadata, rate, MSE, and primary gates.
All 10 lineage tamper cases were rejected.

This is one deterministic precommitted expert panel, not a full-checkpoint,
probability-sample, perplexity, downstream-task, inference, SOTA, or universal
rate-distortion claim. V2 development used the already-opened v1 result; the
confirmatory v2 panel uses disjoint layer/expert coordinates but must not be
pooled with v1. See the result card for the startup-freeze coverage gap,
charged-metadata details, compact-release boundary, and unresolved upstream
polar-code reuse terms.

The historical STRATA blind-v1 result remains a valid negative result:
`0.05166003144302383` pooled relative MSE at `2.14990912543403 bpw`, which is
`1.7615%` above the strict Gaussian reference. It is neither overwritten nor
pooled with v2.

- [STRATA-v2 architecture](docs/STRATA_XKLT_SC_V2.md)
- [Frozen physical format](docs/STRATA_V2_FORMAT.md)
- [Post-run conformance appendix](docs/STRATA_V2_CONFORMANCE.md)
- [Second blind-panel protocol](docs/STRATA_V2_BLIND_PROTOCOL.md)
- [Reproduction and audit guide](docs/STRATA_V2_REPRODUCIBILITY.md)
- [Final independent result and canonical artifacts](docs/STRATA_V2_RESULTS.md)

Verify the compact STRATA release without Qwen source payloads, CuPy, or a
GPU:

```bash
python tools/verify_strata_v2_release.py --repo-root .
```

The 64-file release manifest is
[`release/strata_v2_release_manifest.json`](release/strata_v2_release_manifest.json)
(SHA-256 `dcf87419d6c35b9ba2217e3e1b512dfeb4abfba4841b10761e2d662f91f55bd4`). One
freeze-bound historical base-encoder file is deliberately withheld because it
identifies itself as a direct port of upstream MATLAB code with no visible
license grant. Its exact path and SHA-256 remain bound in the freeze, intent,
manifest, and verifier; see the third-party notice.
Pre-existing POLARIS translation files remain historical evidence under the
same unresolved terms; this repository makes no open-source license grant.

The published POLARIS-SC-v2 result below is preserved unchanged.

This repository is an auditable research implementation of a post-training
weight codec operating below 2.15 bits per weight. The principal result is
deliberately narrow:

> On a deterministic 32-block panel of real BF16 weights from the pinned
> `Qwen/Qwen3-30B-A3B` checkpoint, POLARIS-SC-v2 with a deterministic
> randomized Hadamard transform (RHT) produced an independently decoded,
> energy-weighted relative MSE of **0.0528944847** at an emitted physical rate
> of **2.1497192383 bpw**. This is **4.1932%** above the matched-Gaussian
> rate-distortion limit and therefore inside the declared 5% gate.

The unpreconditioned codec did **not** meet that quality gate on the same
weights. Its MSE was `0.0631987377`, or `24.4908%` above the Gaussian limit.
The positive result belongs to **POLARIS-SC-v2 + deterministic RHT**, not to
the exact unpreconditioned v2 codec.

This is a source-codec/PTQ experiment, not a full quantized Qwen release. The
test covers 32 blocks (8,388,608 weights), not the entire checkpoint, and no
perplexity or downstream task evaluation has yet been run.

## Headline results

| Metric | Exact POLARIS-SC-v2 | POLARIS-SC-v2 + RHT |
|---|---:|---:|
| Blocks / BF16 weights | 32 / 8,388,608 | 32 / 8,388,608 |
| Mean logical payload, bits/block | 559,903.3750 | 562,697.3125 |
| Logical reservoir headroom | 113,940 bits | 24,534 bits |
| Emitted physical rate | 2.1497192383 bpw | 2.1497192383 bpw |
| Energy-weighted relative MSE | 0.0631987377 | **0.0528944847** |
| Excess over Gaussian limit | 24.4908% | **4.1932%** |
| Blocks individually below the 5% ceiling | 27/32 | **32/32** |
| Fresh independent decodes | 32/32 | 32/32 |
| Joint rate-and-MSE gate | **Fail** | **Pass** |

The matched-Gaussian reference at `R = 2.15` is

```text
D_G(R) = 2^(-2R) = 0.050765774772264724
5% ceiling          = 0.053304063510877964
```

The RHT arm reduces MSE by `16.3045%` relative to exact v2 on identical
source blocks. That comparison is paired; it is not a claim of a 16.3%
improvement over every other quantizer.

See [docs/RESULTS.md](docs/RESULTS.md) for the full protocol, rate ledger,
role breakdown, integrity checks, and claim boundary. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for environment setup and exact
commands.

## Architecture

POLARIS-SC-v2 is a six-level Construction-D polar-lattice codec with
randomized successive-cancellation decisions and arithmetic coding. The Qwen
variant completes it with a deterministic, orthonormal RHT in front of the
frozen polar core:

```text
BF16 block x (N = 2^18)
       |
       +-- deterministic signs s from (revision, tensor, block index)
       |
       +-- y0 = H_N diag(s) x / sqrt(N)         [RHT variant only]
       |
       +-- FP64 RMS g of y0 (or x); normalize y = y0/g; store g as FP16
       |
       +-- six-level polar lattice, eta=0.25, alphabet=64, D_test=0.05110
       |
       +-- causal arithmetic bitstream with randomized SC decisions
       |
       +-- checkpoint-global fixed-capacity overflow reservoir
       |
       +-- independent decode and inverse RHT
```

The RHT seed is derived from a canonical block identity and adds zero
per-block side bits under the declared checkpoint/schema convention. It does
not alter the polar quantizer parameters. Because the transform is
orthonormal, it preserves source energy and squared-error distance.

The global reservoir is the other important systems component. Each block
contributes a 32-bit logical payload length and a 16-bit FP16 scale. Variable
arithmetic payloads are concatenated bitwise into one fixed physical payload,
so unusually long blocks can use savings from shorter blocks. Unused suffix
bits must be zero. Rate is therefore checked from an actual serialized file,
not estimated from entropy.

## Why the RHT matters

The frozen polar core was tuned for a unit-variance Gaussian source. Several
real Qwen blocks are strongly heavy-tailed. On this panel, raw Pearson
kurtosis reached `114.2918`; exact-v2 error and raw kurtosis had correlation
`r = 0.9952`. The worst attention-output block had exact-v2 MSE `0.19918145`.

After deterministic RHT, median kurtosis was `3.00146`, the maximum was
`3.01946`, and the worst block's MSE fell to `0.05284034`. The transform makes
the empirical source much closer to the distribution for which the polar
test channel was designed. This mechanism is more informative than the
aggregate alone: exact v2 already fit the bit budget, but failed due to source
distribution mismatch.

## Repository layout

```text
src/
  polaris_sc_v2_encoder.py       frozen, unpreconditioned encoder
  polaris_sc_v2_rht_encoder.py   same polar core plus deterministic GPU RHT
  reservoir_pack_v2.py           fixed-capacity global reservoir writer
  reservoir_unpack_v2.py         strict standalone parser/unpacker
  independent_decoder_v1.py      encoder-independent polar decoder core
  qwen_reservoir_decode.py       BF16/Qwen record decoder and MSE audit

tools/
  build_decoder_map.py           recreate the non-redistributed set map
  fetch_qwen_block.py            fetch one immutable BF16 range
  materialize_qwen_panel.py      fetch/hash-check all 32 manifest blocks
  run_qwen_release.py            convenient published-layout entry point
  run_qwen_panel.py              fail-closed encode/pack/unpack/decode runner
  audit_qwen_paired.py           paired exact-v2/RHT audit
  audit_qwen_distribution.py     CuPy source-distribution audit
  verify_release.py              dependency-free offline release audit
  verify_gaussian_confirmation.py safe pristine-copy Gaussian runner

codec_data/
  README.md                      local decoder-map generation instructions

results/qwen/
  manifest.json                  frozen block identities, hashes, and seeds
  exact_summary.json             unpreconditioned control result
  rht_summary.json               primary RHT result
  paired_audit.json              paired rate/distortion audit
  distribution_audit.json        kurtosis/tail diagnostics
  independent_audit.{json,md}    separate result audit
  independent_audit_erratum.*    local/canonical-index audit correction
  exact_panel.plrsv2             emitted exact-v2 reservoir
  rht_panel.plrsv2               emitted RHT reservoir
  {exact,rht}_pack.json          serializer audits
  {exact,rht}_unpack.json        independent parser audits

results/gaussian/                compact matched-Gaussian confirmation
frozen/gaussian_confirmation/    original-name Gaussian rerun workspace
docs/RESULTS.md                  detailed interpretation and caveats
docs/REPRODUCIBILITY.md          end-to-end reproduction instructions
THIRD_PARTY_NOTICES.md           provenance and redistribution cautions
```

The Qwen checkpoint itself and extracted BF16 source blocks are not
redistributed here. Reproduction fetches the pinned upstream revision and
verifies every extracted block against the frozen manifest hash.

## Quick verification

First run the standard-library release check. It verifies the canonical file
hashes, rederives all 32 seeds, independently reparses both Qwen reservoirs,
and checks the published rate/distortion invariants without a GPU or model
download:

```bash
python tools/verify_release.py
```

You can then validate and independently unpack a
checked-in physical reservoir. This checks its header, directory, logical
payload boundaries, hashes, and mandatory zero suffix without downloading the
checkpoint:

```bash
python src/reservoir_unpack_v2.py \
  --input results/qwen/rht_panel.plrsv2 \
  --output-dir runs/rht-unpacked \
  --audit runs/rht-unpack-audit.json
```

Compare the reported reservoir SHA-256 and rate fields with
`results/qwen/rht_summary.json` and `results/qwen/rht_unpack.json`. Repeat
with `exact_panel.plrsv2` for the control. Recomputing distortion requires the
pinned BF16 source blocks and decoder map; it is covered by the full flow.

For a full rerun, create a CUDA-capable environment, install
`requirements.txt`, clone the pinned polar-lattice repository, generate the
hash-bound decoder map locally, fetch the manifest-selected blocks, and run
both paired arms. The decoder map is deliberately not redistributed because
the upstream reliability tables have no explicit license. The reference run
used Python at
`/workspace/int2-cupy-venv/bin/python`, CuPy `14.2.0`, an NVIDIA RTX 5090,
and four concurrent workers. Follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
rather than treating the abbreviated command above as a complete setup guide.

The runners are intentionally fail-closed. They bind source blocks, seeds,
implementation files, payloads, FP16 scale bytes, reconstruction indices, and
reservoir sizes by hash or exact equality. Existing work directories are not
silently resumed or relabelled.

## What is and is not established

Established by the checked-in artifacts:

- actual BF16 Qwen weights were encoded and reconstructed;
- the same deterministic 32-block panel was used for both variants;
- the RHT result passes the declared 5%-of-Gaussian MSE threshold;
- both emitted panel reservoirs are below 2.15 bpw;
- all 32 records in each arm passed fresh-process decode checks;
- deterministic RHT needs no stored sign vector or learned side model.

The preserved independent-audit JSON contains a documented
`manifest_alignment_ok` comparison bug: it compared tensor-global block
indices with extracted-file-local indices. The corrected mapping passes 32/32
for both variants and does not change any result; see
[`results/qwen/independent_audit_erratum.md`](results/qwen/independent_audit_erratum.md).

Not established:

- a full-checkpoint rate or distortion measurement;
- a probability-sample confidence interval for Qwen weights;
- perplexity, task accuracy, or generation quality after quantization;
- production throughput, kernel efficiency, or deployable inference speed;
- superiority over all current 2-bit quantization systems.

The 32-block panel spans every rank-2 matrix role, but it is a deterministic
coverage panel rather than a probability sample. A role-population expansion
is included in the results as a descriptive diagnostic only. It is not an
estimator with a design-based confidence interval.

## Method provenance

The polar core follows the public polar-lattice construction of Liu, Shi, and
Ling and uses the upstream code pinned at commit
[`458187b9b03db1768a4b72d617e591f7862f6fca`](https://github.com/graceBaoXP/PolarLatticeQuantization/commit/458187b9b03db1768a4b72d617e591f7862f6fca).
The evaluated model is
[`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39)
at revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

This repository should be read as a reproducible experimental artifact. The
next decisive milestone is a preregistered probability sample or complete
`116,470`-block census, followed by full-checkpoint reconstruction and model
evaluation.
