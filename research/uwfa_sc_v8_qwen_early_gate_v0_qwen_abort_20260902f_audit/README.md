# UWFA-SC v8 real-Qwen early gate, run 20260902f

Date: 2026-09-02

Status:

```text
EARLY_DIAGNOSTIC_ABORT_RUNTIME_BUDGET_BEFORE_FIT
```

This is the first successful end-to-end parsing and causal reconstruction of
the pinned Qwen artifact by the repaired exploratory v8 early-gate wrapper.
It is **not** an entropy result.  The producer stopped at its frozen pre-fit
runtime gate, before fitting or scoring any WFA candidate.

## Exact publication

The adjacent directory
`research/uwfa_sc_v8_qwen_early_gate_v0_qwen_abort_20260902f/` is an exact copy
of the completed RunPod publication.  Its member hashes are:

```text
BOUND_BASELINE_SCORE.json
06cc568271a834026afd33e331a8d933644f61ab9c973de1b7eb975229bb3dea
COMPLETE.json
40d8a45af7068140cf698961136b840844cc8a9ec024bd6574a5df43eb14b356
DECODER_BUNDLE.json
d463a50dbb58daff710a04827349ba9846dfac1efe5c47b77ea65d628988c066
RESULT.json
797ebdbe5a6d570e96d82398fa6f19b93fbf0f6973d54c5dd5dd83ffd1477d68
SOURCE_PREFLIGHT.json
65b2cb5b172e83922d87055eb58c13fcdada7caf1bfd90d2d8488025efa058b4
```

`COMPLETE.json` is present and was written last.  The detached launcher exited
zero.  A separately frozen independent verifier is replaying the publication;
until its durable receipt completes, the result-audit status is
`INDEPENDENT_RESULT_AUDIT_PENDING`.

The adjacent launcher receipts are exact copies from RunPod:

```text
uwfa_sc_v8_qwen_early_gate_v0_result_20260902f.log
c7c653f301d1e27234a789c616c4dfd66dd7e79d43f054b6077648908c21bc06
uwfa_sc_v8_qwen_early_gate_v0_result_20260902f.exit
9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
```

## Bound inputs

```text
Qwen artifact bytes       8,847,360
Qwen artifact SHA-256     4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b
runner SHA-256            399cb25260d34ec299cc91a17f129da9be5ba5b799c961e43f0c1b0637ee0174
sealed v8 manifest        a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6
sealed source root        be06cf4d6c474a01517c4062f448b0c41c7f59d31724d6d5af380b8c064de4fa
STRATA common source      3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1
frozen external auditor   85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e
```

The one-time decoded panel was reused by exact object identity:

```text
wrapper extraction calls       2
underlying causal decodes      1
same panel object reused       true
cache receipt SHA-256          2dd4715c9d4bfe01690880492c27a4b976c74b5152b64715963547ef9ca7f34e
full source geometry SHA-256    4ba7afd544b8055e066c0cb859071cef5b9bfb63b807e1d4549f572850bbf67a
reconstruction F64 SHA-256     84309366c3bbc6459d461f1b3e23c48944623ae7dfb8bab7e0c4698f3e661d67
```

## Why the fit did not run

The real panel contains 15 streams and 126,627,266 selected SC symbols.  Its
exact disjoint-component primary plan was estimable, and static host/VRAM
admission passed.  The frozen v8 runtime check nevertheless budgets the
largest possible source-survivor path--primary source plus four shuffles and a
coordinate-disjoint diagnostic--at a fixed floor of one million cell-symbol
updates per second:

```text
disjoint dependence components                    3
exact primary updates                             38,621,316,130
coordinate-disjoint diagnostic updates            93,518,490,096
maximum source-survivor updates                   286,625,070,746
frozen conservative throughput                     1,000,000 updates/s
projected wall                                    286,625.070746 s
projected wall                                     79.6180752072 h
frozen budget                                      21,600 s = 6 h
resource admission                                pass
runtime admission                                 fail
```

Accordingly:

```text
winner                              null
pooled heldout saving               null
physical candidate                  null
controls run                        false
positive-claim authority            false
```

No positive or negative Qwen conclusion about the WFA source law is permitted.
The synthetic `0.16754150390625` bit/symbol fixture result remains detector
calibration only.

## Efficient successor

The authenticated RTX 5090 source-free benchmark measured
`6,484,796.4212` updates/s and froze a 50%-throughput conservative value of
`3,242,398.2106` updates/s.  At those rates the unchanged exact primary plan is
approximately:

```text
measured-rate primary       1.6543531474 h
50%-throughput primary      3.3087062947 h
```

A new source-only v9 sibling therefore keeps the full candidate bank, exact
disjoint-component nested holdout, literal model bytes and final physical
container, but executes only the primary gate.  Survivor-only shuffles,
coordinate diagnostics and controls become separately authorized stages.  A
primary survivor remains nonpromoting; this scheduling change removes work
that is not executed on an early kill rather than weakening the estimand.
