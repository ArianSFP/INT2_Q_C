# STRATA-XKLT-SC v2 blind result

The frozen STRATA-XKLT-SC v2 codec **passed** its preregistered primary gate
on one deterministic, precommitted panel of 18 BF16 expert matrices from
`Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

At an exact route-inclusive physical rate of `2.149999830457899 bpw`, the
independent original-source-domain score was `0.04985939119332436`. That is
`1.7854225272960011%` below the strict Gaussian reference
`2^-4.3 = 0.050765774772264724`, a margin of
`0.07824047385148768 dB`.

This is a result for the named 18-matrix panel and frozen artifact. It is not
a full-checkpoint result, a probability-sample estimate, a perplexity or task
evaluation, an inference benchmark, a state-of-the-art comparison, or a
universal rate-distortion theorem.

## Final independent result

| Metric | Audited value |
|---|---:|
| Model panel | 6 expert triplets / 18 matrices |
| Original BF16 source blocks | 108 × `2^18` values |
| Source weights | 28,311,552 |
| One-shot encoder invocations / retries / resumes | 14 / 0 / 0 |
| Physical bytes / bits | 7,608,729 / 60,869,832 |
| Exact route-inclusive physical rate | 2.149999830457899 bpw |
| Integer cap for `<= 2.15 bpw` | 60,869,836 bits |
| Integer headroom | 4 bits |
| Logical arithmetic payload | 60,768,579 bits |
| Byte-padded payload | 7,596,079 bytes |
| Zero reservoir tail | 7,096 bytes |
| Independent source energy, FP64 | 16,192.89450885593 |
| Independent SSE, FP64 | 807.3678618692818 |
| Pooled source-domain relative MSE | **0.04985939119332436** |
| Gaussian reference at 2.15 bpw | 0.050765774772264724 |
| MSE/reference ratio | 0.98214577472704 |
| Margin below reference | **1.7854225272960011% / 0.07824047385148768 dB** |
| Independent container audit | `passed: true` |
| Audit execution | `audit_execution_passed: true` |
| Primary claim gate | **PASS** |
| Lineage mutations rejected | 10/10 |

The pooled metric is exactly

```text
807.3678618692818 / 16192.89450885593
    = 0.04985939119332436.
```

It is an energy-weighted ratio of FP64 sums over every original BF16 value,
after independent decode, inverse RHT, canonical unordering, and
scaled-orthogonal inverse KLT. It is not an average of the 18 per-matrix
ratios. The encoder-side staged score (`0.049857608894741064`) is diagnostic
and is not substituted for this result.

## Exact physical-rate accounting

The complete `.bin` file is charged; the rate is not an entropy estimate or
an arithmetic-payload-only number.

| Physical section | Bits |
|---|---:|
| Header, including six source-derived KLT codes and coefficients | 1,024 |
| Literal 18-matrix route | 1,152 |
| Raw 3-bit source-derived labels | 41,472 |
| Fourteen profile/scale/length directories | 784 |
| Fixed arithmetic reservoir, including padding and zero tail | 60,825,400 |
| **Total** | **60,869,832** |

Source-derived metadata is not free. The KLT representation, route, literal
labels, profile bytes, FP16 decoder scales, logical lengths, byte padding, and
unused reservoir all occupy and are charged to the physical container. The
exact gate is integer arithmetic:

```text
60,869,832 * 20 <= 43 * 28,311,552
```

The artifact used 60,768,579 logical arithmetic bits. All 14 final-byte
padding checks passed, the byte-padded streams fit the reservoir, and the
remaining 7,096 bytes were independently verified as zero.

## What the independent audit established

The independent auditor:

- parsed the fixed-size physical container and independently recomputed its
  exact rate;
- regenerated the six FP32 KLT coefficient pairs from their Q15-over-pi
  codes;
- authenticated 18/18 matrices and 108/108 nested source blocks;
- rederived the source-dependent KLT staging, eight equipopulous label
  strata, fourteen FP16 scales, exact allocation DP, and all seeds;
- procedurally rebuilt Q31-BEC frozen sets without an external reliability
  map or encoder probability sidecar;
- causally decoded and canonically re-encoded all 14 arithmetic payloads,
  matching every logical length and payload byte;
- reconstructed all 28,311,552 values and scored them against the original
  BF16 sources in FP64; and
- verified the complete selection → source → freeze → allocation → intent →
  summary lineage and all five primary-gate conditions.

The separate tamper harness rejected all ten expected, uniquely named
mutations: selection role, nested source hash, codec threshold, manifest
source binding, allocation profile, retry permission, summary artifact hash,
coefficient regeneration, nonzero stream padding, and directory scale.

## Canonical evidence

| Artifact | SHA-256 |
|---|---|
| [Physical container](../strata_v2_blind_one_shot_v2/strata_xklt_sc_v2.bin) | `e89e2a97fa655cc4248de849ba1b1b84b46e8357044beb563318efb359be2be7` |
| [Independent decode/source audit](../strata_v2_blind_one_shot_v2/independent_audit/independent_decode_audit.json) | `310d04352a6fd57ac8b4cc37f99f17a327a9663c41c0ef2821f2999cc2797d78` |
| [Independent physical inspection](../strata_v2_blind_one_shot_v2/independent_audit/inspection.json) | `ba7d936db86b7679bd7c9f1b741477e1f5e8998679d5e267320540e1cbaabfcf` |
| [Lineage tamper receipt](../strata_v2_blind_one_shot_v2/independent_lineage_tamper_tests.json) | `4122e40542fb7c77780af544fca05d88ff4070aacaa9154b45751b447e594a0c` |
| [One-shot summary](../strata_v2_blind_one_shot_v2/summary.json) | `35770b1ea622c98abc1e76e513fe24446bd14a6bed56e0fb6243b59b9f7a8b87` |
| [Pre-encoding manifest](../strata_v2_blind_one_shot_v2/preencoding_manifest.json) | `5cdced62fe7ccba39e2e313bf40a79820e285fbb86323424a8da697161dd6539` |
| [Allocation lock](../strata_v2_blind_one_shot_v2/allocation.lock.json) | `65efad51f92047236217ff1e90da2fc5cbac567de99ac8ff845a1533e5b1632a` |
| [One-shot intent](../strata_v2_blind_one_shot_v2/ONE_SHOT_INTENT.json) | `0878e9844d6e8249e51e2e4a6586c436a73b9671c5cf3e11ae7cff500c2bf299` |
| [Source-finalization lock](../blind_protocol_v2/unblinded/source_hashes.lock.json) | `bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23` |
| [Selection lock](../blind_protocol_v2/selection.proposal.lock.json) | `528250d8c6bac52dfdf64958d7f4929a115ff68d907a47880cab85d532aade14` |
| [Codec freeze](../blind_protocol_v2/codec_freeze.lock.json) | `1cd70bfc3147663b15039d0f949e5e4b999bf1c6981d33d21dbc4f1b7d7ebda7` |
| [Freeze validation](../blind_protocol_v2/codec_freeze.validation.json) | `9b2472b8ffefd96f0ed36919b0d431818229985820daf82ca1e9c8e5455e8c2d` |
| [64-file release manifest](../release/strata_v2_release_manifest.json) | `dcf87419d6c35b9ba2217e3e1b512dfeb4abfba4841b10761e2d662f91f55bd4` |

The normative format is
[`strata_v2_codec/FORMAT.md`](../strata_v2_codec/FORMAT.md); the independently
executed implementation is
[`strata_v2_klt_mixed_independent_auditor_v1.py`](../strata_v2_klt_mixed_independent_auditor_v1.py).

## Historical v1 and development disclosure

The first blind STRATA artifact remains a clean negative result:

| Metric | Historical blind v1 |
|---|---:|
| Physical rate | 2.14990912543403 bpw |
| Pooled source-domain relative MSE | 0.05166003144302383 |
| Excess over the same Gaussian reference | 1.7615345668824% |
| Verdict | **Fail** |

V2 was designed after that v1 panel had been opened, so the architecture was
informed by v1 and other development evidence. The six v2 `(layer, expert)`
coordinates were disjoint from the v1 panel and were sealed before their
selected BF16 payloads were materialized. This supports the stated second-
panel confirmation only. The two panels must not be pooled, and panel
disjointness does not turn this deterministic selection into a random sample.
The preserved independent v1 receipt is
[`blind_protocol_v2/v1_failure_independent_audit.json`](../blind_protocol_v2/v1_failure_independent_audit.json).

## Limitations and audit caveats

- **Narrow panel:** this is one deterministic, precommitted 18-matrix expert
  panel, not a checkpoint census. It gives no sampling-based confidence claim
  for other experts, layers, roles, models, or data distributions.
- **No model-quality claim:** no full quantized checkpoint, perplexity,
  downstream task, generation-quality, latency, memory-bandwidth, or kernel
  benchmark is reported. Calling this result model-level SOTA would exceed
  the evidence.
- **Source-derived control:** KLTs, labels, profiles, and scales depend on the
  selected weights. They were sealed before arithmetic encoding and are
  physically charged, but this is still a weight-adaptive PTQ codec rather
  than a source-independent quantizer.
- **Startup freeze coverage gap:** the one-shot runner imports Python/NumPy
  and local codec modules before its in-process runtime-freeze verification
  function executes. The freeze and independent audit bind the interpreter,
  package trees, executing files, CUDA identity, and post-run hashes, but they
  are not a cryptographic sandbox or attestation of behavior that could occur
  during process startup/import. The positive claim relies on the published
  source, hashes, artifacts, and independent replay checks within that stated
  protocol boundary.
- **Compact release:** original Qwen BF16 payloads, individual large
  arithmetic sidecars, and reconstructed FP64 arrays are not redistributed.
  Their byte-range provenance and hashes are retained so an authorized user
  can rematerialize and audit them.
- **Withheld historical encoder dependency:** the freeze-bound base encoder
  (`agent_polaris_qwen_rht_encoder.py`, SHA-256 `062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0`)
  identifies itself as a direct port of upstream MATLAB code and is not
  redistributed without a visible license grant. The manifest and verifier
  retain its historical path/hash binding. Artifact inspection and the
  independent audit are included; exact public re-encoding is incomplete.
- **Unresolved reuse terms:** the frozen polar implementation has provenance
  in `graceBaoXP/PolarLatticeQuantization` commit
  `458187b9b03db1768a4b72d617e591f7862f6fca`, for which no explicit upstream
  license was visible when this release was prepared. The research evidence
  is published for auditability, but it does not resolve or grant reuse rights
  in upstream-derived material. See [Third-party notices](../THIRD_PARTY_NOTICES.md).

See the [architecture](STRATA_XKLT_SC_V2.md), [physical format](STRATA_V2_FORMAT.md),
[blind protocol](STRATA_V2_BLIND_PROTOCOL.md), and
[reproduction guide](STRATA_V2_REPRODUCIBILITY.md) for the complete method and
verification flow.
