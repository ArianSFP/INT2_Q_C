# Static review threat model

This review authenticates immutable source bytes and checks the intended
control flow. It treats SHA-256 collision resistance and the pinned producer
manifest as trust anchors.

It explicitly covers:

- mutation, omission, duplication, path traversal, symlink substitution, and
  non-canonical ordering in the flat producer closure;
- bypass of the compile-time capability hold;
- dependency-root confusion, including the coarse auditor's domain-separated
  root algorithm;
- dummy, self-authored, aliased, or semantically unrelated runtime audit
  receipts;
- sample mutation, rank omission, projected candidate scoring, weak-role
  masking, control execution before survival, rate drift, encoder-only score,
  and a page-layout projection mislabeled as an executed trace.

It does not cover:

- correctness of code that was not executed on the final frozen source;
- authenticity of a future deployment sibling before Python begins executing;
- compromise of the capability issuer, external auditors, operating system,
  Python/CuPy/CUDA runtime, GPU, or filesystem after authentication;
- actual Qwen/coarse bytes, numerical results, runtime cost, or inference HBM.

The deterministic sixteen-block-per-role aperture is an early-kill heuristic.
Its bootstrap lower bounds are conditional on treating sampled blocks as an
exchangeable empirical population; they are not probability-sampling bounds.
