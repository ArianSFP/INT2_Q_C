# Qwen early-gate launch 20260902e: pre-payload path failure

This launch is not a model run and contains no scientific result.

The isolated candidate checkout intentionally did not contain the pinned
STRATA common-source directory.  The launcher incorrectly pointed to that
absent local copy and failed during fail-closed path validation:

```text
EarlyGateError: STRATA common source component absent:
/workspace/INT2_Q_C_qwen_v8_20260902e/strata_expert_local_codec
```

The Qwen artifact was not opened.  The empty output directory name was not
created; only the log and exit receipt exist.  The corrected launch uses a new
namespace and the previously authenticated sources at:

```text
/workspace/INT2__compression/strata_expert_local_codec/common.py
  SHA-256 3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1

/workspace/INT2__compression/strata_v2_klt_mixed_independent_auditor_v1.py
  SHA-256 85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e
```

The `20260902e` log and exit receipt remain preserved.  They may not be
interpreted as entropy, rate, distortion or Qwen evidence.
