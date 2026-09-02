# LOGIC-Q prior-art boundary

This is a research-novelty assessment, not a patentability opinion.

## Direct mathematical lineage

- Ye and Abbe, *Recursive Projection-Aggregation Decoding of Reed-Muller
  Codes*, develops efficient projection/recursive decoding and list variants
  for RM codes. LOGIC-Q v0 uses only exact first-order affine enumeration and a
  bounded list; RPA is a future scalable backend, not an implemented result.
  https://arxiv.org/abs/1902.01470
- Fomin, Golovach, and Panolan, *Parameterized Low-Rank Binary Matrix
  Approximation*, records that low-GF(2)-rank approximation is NP-complete even
  at rank one and develops parameterized algorithms. This is why the v0
  alternating factor screen has no global-negative authority.
  https://arxiv.org/abs/1803.06102
- Fomin et al., *Approximation Schemes for Low-Rank Binary Matrix
  Approximation Problems*, gives approximation schemes whose hidden dependence
  on rank is unsuitable for treating rank 680 as an easy exact search.
  https://arxiv.org/abs/1807.07156
- Kieffer, Flajolet, and Yang, *Universal Lossless Data Compression Via Binary
  Decision Diagrams*, losslessly codes power-of-two binary strings through
  ROBDDs. LOGIC-Q differs by changing four-level labels under weighted MSE,
  using an exact mixed-radix coordinate domain, and charging a literal numeric
  quantizer and exception overlay.
  https://arxiv.org/abs/1111.1432
- Khoromskij, *O(d log N)-Quantics Approximation of N-d Tensors in
  High-Dimensional Numerical Modeling*, establishes binary/multilevel index
  folding as a way to expose low tensor ranks. QTT is not implemented in v0;
  ROMDD is the bounded third-family diagnostic.
  https://doi.org/10.1007/s00365-011-9131-1

## What is and is not new here

Reed-Muller codes, soft/list decoding, GF(2) low-rank approximation, decision
diagrams, QTT, entropy-constrained quantization, and exception coding all
exist. None is individually a novelty claim.

The research question not answered by those references is whether a frozen
post-training weight codec can jointly choose nearby legal low-bit labels and a
charged algebraic coordinate function, reproduce numeric SwiGLU-MoE weights,
beat a finite Gaussian reference under raw-weight MSE, and retain expert-local
reads. v0 is only a decisive mechanism/accounting gate for that question.

The strongest future novelty candidate would be the integrated construction:

```text
coarse-decoder-visible geometry
  + label-flexible algebraic/causal codebook
  + exact source-MSE objective
  + posterior centroids
  + literal expert-local packet.
```

No source result is inferred from the mathematical literature.
