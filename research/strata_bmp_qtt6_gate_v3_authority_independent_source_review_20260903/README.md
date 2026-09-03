# Independent source review: STRATA BMP/OBDD/QTT6 v3 authority

Date: 2026-09-03

Reviewed producer:

- manifest SHA-256: `7901e78eaf7c6b854d7bfaa2afbb4eb7be337449a72ef66e66d00adb87f64ab4`
- source root: `14ec1fdc19435f4f3655b4f3458ef774a6503d9c88c2d62c510815499c14aecd`
- exact closure: eight non-manifest regular files

This is an independent benign source-only review. It did not modify the
producer and did not access the network, RunPod, Qwen, model tensors, STRATA
packets, Gaussian controls, reconstruction payloads, or CUDA/CuPy.

The producer's compiled production hold is effective. Its exact capability
closures, separately pinned receipt hashes, explicit fixture flags, literal
source-file alias checks, receipt arithmetic, and event/page-hash checks are
present. Consequently the frozen package cannot currently authorize a payload
run.

The requested semantic claims are not yet established. In particular, the
adapter's scale membership, transform identity and canonical replay are
booleans/hashes in a receipt rather than facts replayed from the packet. The
scorer recomputes formulae from supplied totals but never decodes BF16 or opens
a reconstruction. Weight counts are not tied across source bytes, adapter and
scorer; packet aliases are not rejected; and a trace can omit packet pages.
These gaps could admit a favourable but nonphysical `R/F/read` result after a
launch pin is added.

Run the frozen independent static review with:

```powershell
pwsh -NoProfile -File research/strata_bmp_qtt6_gate_v3_authority_independent_source_review_20260903/review_static.ps1 `
  -Producer research/strata_bmp_qtt6_gate_v3_authority `
  -Output /tmp/strata-bmp-qtt6-v3-static-review.json
```

The checked-in receipt records 33 successful static checks. It is not a
runtime, decoder, scoring, or model-result receipt.

```text
PASS_EXACT_SOURCE_AND_EFFECTIVE_COMPILED_HOLD__BLOCK_LITERAL_REPLAY_SCORING_COUNT_PACKET_ALIAS_AND_TRACE_CLAIMS__HOLD_PAYLOAD_AND_RD
```
