# Independent one-run deployment review

This payload-blind review authenticates deployment manifest
`a969382640ad69ee71b6029d901d7eade7b88112d582059d83b947e33d1767c3`
and root `edea8361c0c6d990b9875e0e016e5d31c9cfe525d8803ce2f4d406a2077adae6`.

Verdict: `PASS_AUTHORIZE_EXACTLY_ONE_QWEN_RUN` under the conditions recorded in
`AUDIT_RECEIPT.json`. The copy is byte-identical to repaired source manifest
`b92d4b5f...` except for the sole AST literal
`PAYLOAD_EXECUTION_ENABLED = False` becoming `True`. Internal panel, core, and
worker pins remain exact. Authorization, CUDA-device, payload-root, and output
guards precede payload-worker access, while output creation uses exclusive `x`
mode. No Qwen payload was opened during this review.

The code has no persistent invocation counter. “Exactly one” is the authority
granted by this sealed review: retire the deployment after its first invocation,
irrespective of whether the scientific result passes or fails.
