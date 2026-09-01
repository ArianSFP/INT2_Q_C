# Independent source/reproducibility audit: lossy-tail v8

Status: **PASS_V8_INDEPENDENT_SOURCE_AUDIT**.

This audit authenticates the exact 17-file producer package and exact
11-member launch stage at launch-manifest SHA-256
`6c5f5cd05973dbc0bf16cd9ea39951e690b15e15e13e969d2a33823117c2aa94`.
It independently checks the frozen target, physical-rate, rate-ledger,
strict-read, matched-control tolerance, finite-only, numeric-boundary, and
decision equations. It also replays the documented source-only verifier,
CPU tests, Linux-only compatibility tests, and isolated-stage audit on the
pinned Linux interpreter.

The verdict warrants only the separately approved source-free runtime
calibration step described by the package. It does not authorize Qwen/model
or validation-data access, CuPy/CUDA/GPU use by this audit, production, or any
compression claim. The runtime calibrator's future receipt remains explicitly
untrusted until a separate independent runtime audit.

No model/Qwen directory was traversed, no payload or validation file was
opened, and this audit did not import CuPy or initialize CUDA/GPU. SSH/SCP was
used only to copy the sealed source package and run source-only checks in the
provided RunPod scratch directory; the audited programs made no external-data
network calls.

## Verify

From this directory in PowerShell 7:

```powershell
pwsh -NoProfile -File .\verify_audit.ps1
```

The audit receipt uses the canonical-JSON internal seal required by the v8
authorization contract: remove `audit_receipt_sha256`, serialize with sorted
keys and compact UTF-8 JSON, and SHA-256 the result. The replay receipt and
audit manifest use the same construction with their respective seal fields.

