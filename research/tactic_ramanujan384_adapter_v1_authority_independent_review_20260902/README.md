# Independent review: TACTIC Ramanujan-384 authority v1

Date: 2026-09-02

Producer: `research/tactic_ramanujan384_adapter_v1_authority`

Pinned producer manifest:
`f4ba72b9371d77ad4347d5a4fe377677473844dd696032e662acc6cd3bde22b4`

Pinned producer root:
`6840b6a0eb4f2856f84c610ba11888382ecca257d88ebda7f5b49c0de9f3b3c5`

## Disposition

`SOURCE_REPAIRS_SUBSTANTIALLY_CLOSE_V0__HOLD_PAYLOAD_FOR_RUNTIME_SCALABILITY_AND_COARSE_DECODER_AUDIT`

This is a benign source-only correctness and reproducibility review.  It did
not open model, coarse-model, or matched-model-control payloads and did not use
the network.  The producer was not modified.

V1 materially repairs every correctness-authority defect identified in the
v0 audit.  Its canonical source root is correct; every representable rank
candidate is packet-replayed before valid-coordinate FP64 SSE comparison; the
emitted container is written, read, coarse/fine decoded to literal weights,
and independently FP64 rescored; actual input and auditor manifests are
opened and hashed; tail padding is excluded from MSE; and measured file-read
events are clearly separated from layout and HBM claims.

Payload authority nevertheless remains on hold because source-only and CuPy
runtime receipts are absent, the real coarse decoder remains an external
audited capability, and the candidate implementation has a serious scaling
risk.  The dictionary and Gram matrix are shared, but per-candidate decoding
still dynamically loads the v0 packet module, allocates device arrays, runs a
small solve and dictionary matmul, and synchronizes the host for every SSE.
At Qwen geometry and nine controls this means hundreds of thousands of Python
dispatches/device synchronizations.  It is correct source mechanics, not yet
a credible production encoder.

## Findings

### Closed from v0

* Independent canonical recomputation of all sixteen member rows produces the
  recorded root `6840...`; the manifest SHA-256 is `f4ba...`.
* A composite is replayed from a new regular file.  Coarse FP32 output is
  bound to independently recorded per-role hashes, fine records are parsed
  from the replayed container, and literal weights are compared byte-for-byte
  with the encoder reconstruction before FP64 MSE and `F` are reported.
* Candidate ranks 0--14 are canonical packet encoded and decoded before SSE
  can select them.  The winning concatenated stream is decoded again.
* SplitMix64 plus fixed integer lanes generates host FP64 control bytes without
  NumPy/CuPy random APIs.  Identical host bytes are copied to either backend.
* Actual input and auditor source manifests are mandatory paths and are
  regular-file read, strict-JSON parsed, and hash-bound to the binding and
  receipt.
* The Qwen ledger remains 1,470,464 bytes, 359 pages and exactly `359/144`
  bpw.  Tail packets are charged; only valid source coordinates enter SSE.
* One instrumented data-returning `read(2)` event is recorded separately from
  an unmeasured 1x contiguous-layout bound.  Physical storage and HBM claims
  are explicitly denied.

### Remaining gaps and boundaries

1. **Runtime receipts are absent.**  Local Python was unavailable when this
   review was frozen.  The source tests and independent CPU/CuPy runner must
   execute before payload launch.
2. **Only immutable basis state is shared on CuPy.**  Dictionary and Gram are
   shared.  Candidate corrections, support arrays, solves, packet imports and
   scalar SSE synchronizations are not.  This is the primary engineering
   blocker.
3. **The bundled CuPy smoke does not exercise controls.**  Its 64x64 fixture
   page-rounds to more than 2.5 bpw, so the adapter returns at the rate gate.
   The separate runner in this review directly checks control bytes and
   CPU/CuPy packet results.
4. **The controls are Gaussian-like, not Gaussian.**  Twelve summed 16-bit
   uniforms have Irwin--Hall excess kurtosis rather than an exact normal law.
   Backend identity is improved, but scientific reports must not label these
   exact Gaussian controls.  A portable normal sampler or both control
   families should be used for the sealed experiment.
5. **The coarse decoder is a semantic capability.**  Its output must match
   independent FP32 hashes, which closes reconstruction scoring, but v1 does
   not authenticate the decoder implementation itself.  A real pilot still
   needs the separately frozen decoder and receipt named in the design.
6. **Declared extreme geometry exceeds inherited header fields.**  The shape
   contract can accept dimensions whose block count is `2^32`, while the v0
   container encodes each block count as unsigned 32-bit.  Realistic SwiGLU
   shapes are unaffected; the formal universal domain should cap block counts.
   Its page rounding should also use integer arithmetic throughout at the
   declared `2^63-1` total-value bound.
7. **Producer verification is not an independent closure verifier.**  It
   loads `manifest.py` from the package before authenticating that member and
   does not reject extra directory entries.  The fixed manifest hash and this
   review's independent verifier close the frozen snapshot, but producer-side
   verification can be hardened.
8. **Tail coordinates influence candidate generation.**  They are correctly
   excluded from SSE, but zero padding still enters correlation normalization
   and the full-block Gram solve.  This is conservative and cannot create a
   false MSE pass, though it may degrade tail-shape performance.

## Commands

```bash
python -I -B research/tactic_ramanujan384_adapter_v1_authority_independent_review_20260902/test_independent_source.py

python -I -B research/tactic_ramanujan384_adapter_v1_authority_independent_review_20260902/verify_review.py \
  --package research/tactic_ramanujan384_adapter_v1_authority_independent_review_20260902 \
  --manifest-sha256 <review-manifest-sha256>

python -I -B research/tactic_ramanujan384_adapter_v1_authority_independent_review_20260902/run_cupy_reproducibility.py \
  --authorization AUDIT_SOURCE_FREE_TACTIC_RAMANUJAN384_V1_CPU_CUPY_REPRODUCIBILITY \
  --producer-manifest-sha256 f4ba72b9371d77ad4347d5a4fe377677473844dd696032e662acc6cd3bde22b4
```

The CuPy runner accepts no payload path.

