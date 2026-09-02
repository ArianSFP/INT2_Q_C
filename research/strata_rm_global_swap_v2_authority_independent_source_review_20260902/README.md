# Independent source review: global STRATA RM swap v2 authority

Date: 2026-09-02

This is a benign, source-only review of producer manifest SHA-256
`1f1caf2884a8b0b8713f213a16a0a32194238b64969e9d9cf3aaa339ddb776be`
and source root
`e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9`.

The producer was not modified. No model, checkpoint, packet, tensor, Qwen
payload, or Gaussian-control payload was opened. The restored RunPod endpoint
was still refusing the SSH connection when the review began; the parent then
directed this review to stop retrying the network and freeze a static review.

`REVIEW_ASSESSMENT.md` is the normative assessment. `review_static.ps1`
independently checks the exact flat closure, member bytes and hashes, canonical
member-root construction, the externally pinned manifest hash, and a bounded
set of source invariants corresponding to the nine v1 findings. It deliberately
does not execute the producer's Python/CuPy workers.

## Local static execution

```powershell
pwsh -NoProfile -File .\review_static.ps1 `
  -Producer ..\strata_rm_global_swap_v2_authority `
  -Output .\STATIC_REVIEW_RECEIPT.json
```

The checked-in receipt is a static-source receipt only. It cannot authorize a
payload run or a rate-distortion claim.

## Disposition

```text
PASS_STATIC_CLOSURE_AND_SUBSTANTIAL_V1_REPAIRS__BLOCK_PHYSICAL_AUTHORITY_ON_SCIENTIFIC_PROVENANCE_AND_ROUTED_EXPERT_IO__HOLD_PAYLOAD_AND_RD
```

