# TACTIC-CAGE suggestion review supplement

Date: 2026-09-02

Scope: universal weight-only PTQ for SwiGLU-MoE expert triplets, original-weight
MSE, `2.15 <= R_physical <= 2.5`, and routed cold reads below `2x`.

## Verdict

TACTIC-CAGE is a coherent conditional successive-refinement architecture, but
it is not yet a Qwen result. Its strongest open ideas are:

1. use the paid coarse message to select the geometry of the conditional fine
   codebook; and
2. use one non-local conditional source law for both exact fine-symbol
   probabilities and posterior-centroid reconstruction.

The graph, lifting transform, adaptive tree, syndrome, and bits-back backend do
not create the missing below-Gaussian information. They can only expose,
search, or physically realize a conditional source advantage that must first
be measured on held-out model components and charged in one literal packet.

## Information-theory corrections

Let `C` be the paid coarse message, `S` public shape/role semantics, `theta` a
charged model, and `G=Phi(C,S,theta)` a graph or transform. Then

```text
H(G | C,S,theta) = 0
I(X ; G | C,S,theta) = 0.
```

`G` costs no separate descriptor, but supplies no residual information beyond
`C`. It can still be valuable by organizing a source-matched conditional
codebook.

For a 4,096-weight block with a fixed 384-bit fine field, a prefix-adaptive
tree has at most `2^384` leaves per coarse word. A 384-bit syndrome has at most
`2^384` values. Curved geometry and efficient search may help, but neither
construction increases the message alphabet.

Gaussian sources under squared error are successively refinable. A hierarchy
can attain the Gaussian rate-distortion curve but cannot move below it without
non-Gaussian source structure. See [Equitz and
Cover](https://isl.stanford.edu/~cover/papers/paper94.pdf).

The proposal's side-cost table is not a deployed fixed-cap calculation. At a
hard `2.5 bpw` cap, new model or side bytes must displace coarse or fine bytes,
and the resulting distortion must be measured again.

### Posterior reconstruction

For the complete decoded message `M`, the optimal squared-error decoder is
`mu(M)=E[X|M]`, and

```text
E ||X-Y(M)||^2
  = E ||X-mu(M)||^2 + E ||mu(M)-Y(M)||^2.
```

This is a real zero-extra-index opportunity when the current reconstruction is
not the conditional centroid. It is not free model adaptation: any parameters
fitted to the emitted source must be serialized and charged, while discovery
and model-family selection must hold out whole experts/layers or disjoint model
families. The operational score is one joint quantity:

```text
Delta s_joint
  = (R_baseline - R_new) - 0.5 log2(D_new / D_baseline).
```

Rate and centroid gains from separately fitted experiments may not be added.

### Syndrome and bits-back

Because both encoder and decoder possess the same paid coarse message, a fine
syndrome is an implementation of ordinary conditional rate-distortion, not a
new Wyner-Ziv information source. It is blocked until a held-out conditional
law beats a direct conditional tree under the same bytes.

Bits-back ANS can marginalize latent states while losslessly coding an already
chosen discrete label stream. It does not directly refund the entropy of a
lossy stochastic encoder whose posterior depends on the unavailable original
weights. In deterministic PTQ there are no stochastic-encoder bits to recover.
It must also pay an expert-local initial reservoir, termination, finite
precision, posterior mismatch, padding, and model bytes. See
[BB-ANS](https://arxiv.org/abs/1901.04866),
[Bit-Swap](https://arxiv.org/abs/1905.06845), and the explicit lossy-coding
discussion in [Multi-Sample Training for Neural Image
Compression](https://arxiv.org/pdf/2209.13834).

## Actual evidence versus proposed work

| Branch | Current evidence | Disposition |
|---|---|---|
| Source-free chi=64 WFA witness | `0.16754150390625 bit/symbol` after its 3,456-byte model on a synthetic long-parity process | Detector calibration only; it opened no Qwen payload. |
| Real-Qwen WFA/CTW census | V8 decoded 15 streams / 126,627,266 selected SC symbols, then aborted before fit on its declared runtime gate | No positive or negative Qwen WFA result yet. V9 primary-only run is in progress. |
| Real `307/128` coarse stream | V3 specifies byte allocation but no executable packet grammar or independent decoder | Blocked; no Qwen coarse residual exists. |
| TACTIC-DH384 | Synthetic CPU/CuPy arithmetic agrees; the finite 384-bit QC/trellis and scale remain undefined | Not tested on Qwen. |
| Non-local posterior centroid | No latent WFA/HMT posterior has been run | Open. Local RAVEL evidence is adverse but does not dominate a non-local posterior. |
| Coarse graph/Krylov | No run can be valid without a real coarse artifact | Open. Related fixed tangent/local path bounds are weak but non-containing. |
| Adaptive 384-bit tree | Not implemented | Open only after a coarse geometry survives a containment oracle. |
| Coarse-residual syndrome | Not implemented | Open only after a conditional source law survives. Existing aligned-role syndrome opportunity is only `0.0037082566 bpw`. |
| CYCLO-FRI4-NORMAL | Real Qwen dominant oracle reached best `F=0.9379899308`, not `0.8` | Hard kill for its declared period/rank family. Do not rerun the generic cell. |
| TTN/HMT/parity/type hierarchy | Production packages were blocked before payload; parity evidence is synthetic only | Open, but requires the same literal-rate and held-out controls as WFA. |
| Orbit/common stream | Same-layer relaxed Qwen oracle captured only about `0.01553`, versus about `0.14566` required by that screen | Adverse; deprioritize unless a compact exact-orbit generator changes the source decomposition. |

The synthetic and production units must not be conflated. Qwen has
`4.472635975590459` selected SC symbols per source weight. At the strict
page-aligned same-reconstruction ceiling, the production stream needs a
model-charged saving of at least `0.03441710571244585` bit per selected symbol.
The synthetic `0.16754 bit/symbol` result demonstrates detector sensitivity;
it does not transfer that saving.

## Routed-read boundary

The frozen TACTIC layout has a one-pass storage read of

```text
(6 global pages + 359 expert pages) / 360 owner-share pages
  = 73/72
  = 1.0138888889x.
```

A second compressed expert-frame fetch gives

```text
(6 + 2*359) / 360 = 2.0111111111x
```

and fails the strict `<2x` requirement. An expert-wide graph must therefore
decode coarse records once, buffer the required state, and read the fine
stream without refetching the compressed frame. Storage reads, resident HBM
traffic, scratch, graph construction, and inference energy must be reported
separately; a unique-page union alone does not prove one-pass execution.

## Universal-codec boundary

The decoder may depend on public SwiGLU role/shape semantics and on parameters
inside the charged packet. It may not depend on Qwen identity, layer/expert
lookup tables, public reference checkpoints, or uncharged fitted constants.
A universal encoder may fit an allowed model to a new checkpoint only if the
complete fitted state is serialized and charged. Discovery should select the
procedure on disjoint experts/layers and establish portability on a disjoint
SwiGLU-MoE family.

## Decisive execution order

1. Finish the exact primary real-Qwen WFA fit and physical-container test.
2. If its source law survives, add a separately frozen continuous-emission
   head and jointly score operational rate plus posterior-centroid MSE.
3. Independently implement and decode a real universal `307/128` coarse
   packet. Byte allocation alone is not a codec.
4. Run two separate optimistic Qwen bounds: frozen DH384 and a
   coarse-derived graph/Krylov containment oracle. A DH384 failure kills only
   that span.
5. Promote a surviving geometry to an exact dyadic lifting implementation and
   compare fixed frame, nonadaptive graph tree, and prefix-adaptive tree at the
   same 384-bit budget.
6. Consider a syndrome only after measured conditional rate-distortion beats
   direct coding. Consider bits-back only when explicit latent-state cost is
   the measured blocker.
7. Emit one independently decoded packet with `F<=0.8`, physical rate in
   `[2.15,2.5]`, maximum owner-aware cold read below `2x`, and all model,
   transform, framing, padding, tail, and termination bytes charged.

## Current execution checkpoint

The exact primary-only Qwen WFA gate is committed at `9ef874f`. Its source
manifest is
`d1e3eaff6762df2e273f6e3f4216ff9110abe74a7534a0098544a4ceef632c5e`;
the runner is
`d1ff04ce3c2cc36208e464eaed943d6c94eb91a47e9d3c460b2d562b7162cc4d`.
The independent source audit passed it for one nonpromoting primary-only
diagnostic and blocked every positive, universal, or production claim from
that run alone.

The detached RunPod run uses CuPy on the RTX 5090 and writes to
`/workspace/uwfa_sc_v9_qwen_primary_gate_v0_result_20260902_9ef874f`.
Its conservative GPU-kernel projection is `11,911.3426609967 s`
(`3.3087062947 h`), excluding source decode, host physical coding, causal
replay, and publication.

## Detailed evidence

- [Original TACTIC-CAGE assessment](TACTIC_CAGE_RESEARCH_ASSESSMENT_2026-09-02.md)
- [Five-document information/accounting review](../research/tactic_cage_review_20260902/REVIEW.md)
- [Posterior gate](../research/tactic_cage_review_20260902/POSTERIOR_GATE_SPEC.md)
- [Coarse closure gate](../research/tactic_cage_review_20260902/COARSE_CLOSURE_SPEC.md)
- [V8 real-Qwen runtime-abort audit](../research/uwfa_sc_v8_qwen_early_gate_v0_qwen_abort_20260902f_audit/README.md)
- [V9 primary-only runner](../research/uwfa_sc_v9_qwen_primary_gate_v0/README.md)
- [V9 independent source audit](../research/uwfa_sc_v9_qwen_primary_gate_v0_independent_audit_20260902/AUDIT.md)
- [Blocked actual-coarse v3 audit](../research/tactic_actual_coarse_n18_v3_independent_audit_20260902/AUDIT_REPORT.md)
- [Blocked DH384 v2 audit](../research/tactic_conditional_dyadic_coset_v2_independent_audit_20260902/AUDIT_REPORT.md)
