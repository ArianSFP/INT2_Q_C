# Independent source-only audit: UWFA-SC v9 primary gate

Date: 2026-09-02

Audited package: `research/uwfa_sc_v9_qwen_primary_gate_v0`

Payload policy: no Qwen, BF16, Gaussian-control, or other model payload was opened, statted, hashed, enumerated, decoded, or launched by this audit.

## Verdict

```text
PASS_FOR_ONE_EXACT_NONPROMOTING_PRIMARY_ONLY_QWEN_DIAGNOSTIC
BLOCK_FOR_ANY_POSITIVE, UNIVERSAL, OR PRODUCTION CLAIM
```

I found no source-level blocker to launching the exact reviewed v9 package for its declared purpose. The package is a bounded runtime envelope around the byte-sealed v8 science. It runs the unchanged all-150 candidate primary nested holdout and, regardless of whether that primary wins, constructs and independently decodes the unchanged literal v8 final container. It cannot run survivor shuffles, the coordinate-disjoint diagnostic, or matched controls, and a primary survivor is explicitly nonpromoting.

This verdict does **not** say that the WFA works on Qwen. The completed v8 Qwen attempt aborted before fitting because its combined source-plus-diagnostics runtime projection exceeded budget. No Qwen entropy saving was measured. V9 exists to execute the previously unmeasured primary source estimand only.

## Exact source inventory

The v9 manifest declares exactly four members, and the directory contains only those four members plus the manifest. All declared byte counts and hashes matched.

| File | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 4,288 | `b70d5115ed10d1588b074a1536e2c748f6a7fcb71db9393a8df948e636f361eb` |
| `design_lock.json` | 4,506 | `3ca49b26bc41ab0100494085139ab35d20b4be5ea4e75949cb75f089ea7ebbf7` |
| `primary_gate.py` | 37,864 | `d1ff04ce3c2cc36208e464eaed943d6c94eb91a47e9d3c460b2d562b7162cc4d` |
| `test_source_only.py` | 13,984 | `1c272b40c95208c91f05d463d7de85d37604e405114d270163c79bffc7db6964` |
| `SOURCE_MANIFEST.json` | 1,186 | `d1e3eaff6762df2e273f6e3f4216ff9110abe74a7534a0098544a4ceef632c5e` |

External pins and relevant sealed-v8 members also matched the reviewed workspace:

| Input | Bytes | SHA-256 |
|---|---:|---|
| sealed-v8 manifest | 3,518 | `a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6` |
| repaired v8 Qwen support | 41,309 | `399cb25260d34ec299cc91a17f129da9be5ba5b799c961e43f0c1b0637ee0174` |
| sealed `stage0_census.py` | 123,776 | `7b7c2e0fcb6593805e6b2c8234ae59cb42d90fbb7dcf945a35aa5dfe331ae618` |
| sealed `container_codec.py` | 93,379 | `645debb547a76818a880bfc346a2dd6230af97b07dc832afb3548a83d6920fed` |
| sealed `uwfa_common.py` | 58,875 | `db53567ab6d71d5150cc92ef4a78fa9ce5cca01f5474fa2ca32edc8711cc4325` |
| sealed `protocol.py` | 21,051 | `9e18675a1e646eb10c0900aa3767bff96666943309dbd8db3953c745888d2cc1` |
| sealed `strata_sc_adapter.py` | 36,184 | `08fc8808ac168f6930ee9482e160f25f2bd087829fca4630553aea3510d722c6` |
| pinned STRATA common source | 14,320 | `3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1` |
| pinned frozen auditor source | 116,835 | `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e` |

The sealed-v8 manifest has 17 declared members. An independent manifest replay found zero missing members, byte-count mismatches, or digest mismatches.

## Scope equivalence to sealed v8

The only scientific execution path is:

1. `prepare_backend_cache` on the exact authenticated panel;
2. sealed-v8 `nested_holdout(..., policy="exact_identity", diagnostic_only=False)`;
3. select the canonical candidate named by the nested-fold vote;
4. sealed-v8 `final_container(...)` with that candidate.

The v9 runner does not reimplement fitting, likelihood scoring, candidate selection, arithmetic coding, model serialization, packet construction, routed decoding, or physical metrics. These remain in the authenticated v8 snapshots.

The exact candidate bank is checked to contain 150 selectors in canonical order `0..149`. The projected Qwen geometry must have 15 streams, 126,627,266 selected symbols, three disjoint owner-dependence components, and the exact per-fold work below:

| Component | Identity indices | Updates |
|---:|---|---:|
| 0 | 0, 1 | 12,865,688,966 |
| 1 | 2, 3 | 12,794,875,916 |
| 2 | 4, 5 | 12,707,496,716 |

Independent arithmetic reproduces the admitted primary work:

```text
fold sum                         38,368,061,598
final fit plus exact score         253,254,532
                                  ---------------
exact primary                    38,621,316,130 updates
```

No v8 `source_phase` call is made because that entry point deliberately aborts under the original combined-work projection. V9 invokes the two unchanged scientific primitives directly only after reproducing and pinning the original v8 projection, confirming that the full-survivor projection fails, and separately admitting the exact primary work.

## Source-access and provenance review

The CLI path is fail-closed in the reviewed threat model:

- wrong authorization fails before source authentication or path access;
- artifact and output leaf paths receive lexical validation only at the start;
- the v9 package, repaired support, sealed-v8 manifest and all v8 members are authenticated before numerical imports;
- NumPy/CuPy and the two external decoder sources are loaded before the source-free GPU review, but the Qwen artifact remains unopened;
- all-150 and representative source-free receipts are validated against the sealed-v8 validators and an independently queried GPU identity;
- only then does the sole artifact-opening function receive the exact review record and open a no-follow, size- and SHA-256-pinned regular file;
- the artifact is decoded once, and the exact same panel object must satisfy the second extraction request;
- the output directory is created only at publication; members use exclusive creation and `COMPLETE.json` is written last.

The top-level v9 dispatcher is necessarily anchored by this independent review and the exact manifest hash above; its manifest is not pinned by a still-higher in-process launcher. Launch authorization therefore applies only to the exact reviewed Git/source state. A coherently replaced runner plus coherently replaced self-manifest is outside the in-process self-authentication boundary.

The `SourceFreeReview` object uses a public SHA-256 integrity checksum, not a secret or unforgeable capability. That is sufficient for the closed CLI call graph, which constructs the object internally after the real preflight. It must not be treated as an authorization primitive for arbitrary third-party Python callers importing the module.

## Bridge ABI

The pinned STRATA helper returns 15 one-dimensional NumPy integer arrays. The repaired support converts each row to `list[int]`, preserving value and order. This is the correct semantic bridge for the sealed adapter: a Python list performs one-axis advanced indexing, while a tuple of integers would be interpreted as one index per array axis.

The support hash is exactly the previously independently tested bridge. That earlier source-only bridge audit covered all 15 real helper rows and all 13,824 ordinals and established exact selection equivalence. V9 additionally requires exact built-in `int` values, list rows, the pinned support hash, and one underlying artifact decode. The bridge remains exploratory and has no positive claim authority.

## Runtime accounting

At the authenticated reference conservative throughput,

```text
c = 3,242,398.2106118356 updates/s
T = 38,621,316,130 / c
  = 11,911.3426609967 s
  = 3.3087062947 h
```

The code accepts conservative throughput only in `[1,800,000, 4,500,000]` updates/s and recomputes the exact primary projection. At the lower bound the projection is 21,456.286739 s, leaving only 143.713261 s under the 21,600 s kernel-work budget.

This is honestly labeled a GPU kernel-work projection, **not** a total wall-time estimate. It excludes the authenticated panel decode, host arithmetic encode/decode and canonical rebuild, routed/standalone causal decode, physical metrics, and publication. There is no runtime timeout after admission. A slower real workload can therefore exceed six wall-clock hours without creating a scientific-integrity failure; it is an operational risk that must be monitored in the detached launch.

The original maximum 286,625,070,746 updates and the 93,518,490,096-update coordinate diagnostic are recorded but not admitted. No control/shuffle CLI option or call path exists.

## Model, rate, and routed-read accounting

The literal physical result remains the v8 container, not an estimated entropy number:

- the full-panel model is serialized into `model_packet` and charged once in the actual container;
- all streams are causally encoded and decoded using the deserialized transmitted model;
- semantic packet, immutable state, model, directory, frame/region headers, owner contribution records, payloads, alignment and rate-floor padding are included in the byte ledger;
- the container must parse and canonically rebuild byte-for-byte;
- standalone decode must reproduce the pinned FP64 reconstruction digest;
- rate is computed from actual container bytes and source weights and must lie in `[2.15, 2.5]` bpw;
- `F` uses that actual rate and the independently bound same-reconstruction MSE;
- routed cold read uses an authenticated descriptor-backed reader, exact requested ranges and unique touched pages, and the strict maximum of total-physical and nonpadding amplification; it must be below `2x`.

V9 publishes both the literal `UWFCV8.bin` and the identity-framing counterfactual, but only the former drives the physical verdict. The full non-payload physical record is retained in `RESULT.json`; the compact summary does not replace it.

## Nonpromotion and deferred work

Even if all primary gates pass, the strongest possible status is:

```text
PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED
```

Every result and completion envelope fixes:

```text
positive_claim_authority = false
controls_run = false
shuffles_run = false
coordinate_disjoint_diagnostic_run = false
```

A survivor still requires separately reviewed survivor shuffles, a coordinate-disjoint diagnostic, matched Gaussian controls, and a fresh independent result audit. It is not evidence for a universal SwiGLU-MoE codec by itself.

## Source-only test result and coverage assessment

I reran the exact frozen test file on the provided RunPod with the isolated interpreter and bytecode disabled:

```text
/usr/bin/python3.12 -I -B .../test_source_only.py
Ran 12 tests in 0.035s
OK
```

The tests meaningfully enforce import inertness, authorization ordering, review-integrity rejection, exact update pins, fold weakening rejection, canonical all-150 closure, the exact nested-holdout/final-container call scope, absence of control/shuffle entry points, nonpromotion, list-of-list integer bridge semantics, single-decode cache behavior, and package/support authentication.

Their limits are important:

- they are unit/static tests with synthetic objects, not an end-to-end sealed-v8 GPU run;
- they do not open or decode the Qwen artifact and therefore cannot establish a Qwen saving;
- they do not independently exercise every negative branch of publication, no-follow input handling, physical accounting, or routed IO;
- the local synthetic bridge case is weaker than the prior dedicated real-helper source-only bridge audit;
- the runtime projection is verified arithmetically but cannot guarantee total wall time.

Those limits do not block this explicitly nonpromoting primary diagnostic because the runner authenticates and delegates to the already audited sealed-v8 implementation and performs the full source-free GPU review before payload access. They do block promotion from this run alone.

## Launch conditions

The PASS verdict remains valid only if all of the following hold:

1. launch from a clean checkout containing the exact hashes in this report;
2. use `python -I -B` and the exact nonpromoting authorization token;
3. use the exact pinned support, sealed-v8 package, STRATA common source, frozen auditor and artifact paths;
4. use a new absent output directory and preserve detached log plus exit status;
5. treat any primary survivor as a hypothesis requiring the deferred controls, never as a compression result;
6. independently audit every published member and causally replay the literal container before reporting even the primary diagnostic.
