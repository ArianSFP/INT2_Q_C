# Sparse unifilar WFA entropy census v1 — sealed source only

This package implements the smallest decisive operational test of the proposed
MPS/copula branch that is practical at current expert scale. It asks whether a
frozen bank of sparse, causal, deterministic-state probability models can
losslessly recode the **already selected** POLARIS/STRATA binary arithmetic
decisions enough to move the unchanged reconstruction from
`F=0.9888693569009007` to `F<=0.8`.

The exact standalone requirement is:

```text
-0.5*log2(0.8) - 0.008074080480766676
  = 0.15288996696291447 physical bits/source-weight.
```

The source is sealed without payload authority. At seal time no Qwen or other
model payload, current finite artifact, selected-bit array, Gaussian control,
NumPy array, CuPy context, CUDA context, or GPU job was opened, statted, hashed,
enumerated, imported, or launched. A numeric run is forbidden until a separate
agent independently audits this exact source-manifest hash and issues a bound
review receipt.

## What v1 fixes

The prior dense-HMM proposal had both a computational and an evidentiary
problem. A dense chi-state forward belief costs at least `O(chi^2)` per symbol,
and full-panel model selection was too easy to confuse with heldout evidence.
This v1 package replaces it with 150 frozen unifilar cells:

- five procedural topologies: bounded suffix, XOR/parity sketch, modular ones
  accumulator, rolling affine sketch, and saturating signed-count state;
- state sizes `chi={2,4,8,16,32,64}`;
- exact resets at `{32,128,512,2048,4096}` symbols;
- one integer state update and one Q0.16 table lookup per symbol;
- a literal two-byte topology selector and every fitted uint16 table value
  serialized in the model packet.

The fixed bank costs `O(150*N)=O(N)` in stream length and never contracts a
dense chi-by-chi matrix. It is a finite weighted-automaton census, not a claim
to cover arbitrary MPS, Born machines, TTNs, MERA, or a source-coordinate
quantizer-label copula.

## Exact t=0 law

At each canonical arithmetic-stream boundary, and immediately before every
position divisible by the selected reset length:

1. set state `s=0` and within-reset position `t=0`;
2. construct the decoder-visible context from polar level, the 16-bin bucket
   of the regenerated current Q0.16 probability, and `t mod 4`;
3. look up `p(y=1|s,c)` and arithmetic-code the current bit;
4. only then apply the exact symbol-conditioned transition.

There is no learned state transition. The only fitted values are exact integer
counts with Jeffreys half-count smoothing, deterministically rounded to Q0.16.
Layer, expert, model, checkpoint, stream, site, weight, and future-symbol
identity are forbidden probability keys.

## Three distinct evidence objects

The runner never conflates these objects:

1. **Nested scientific fold replicas.** For an outer `(layer,expert)` test
   cell, development excludes every stream sharing that layer **or** expert.
   A frozen SHA-256 rank split assigns a nonempty 20% of development to inner
   validation and the remainder to inner train.
   All cell selection uses inner validation only; the selected cell is refit on
   development and exact-arithmetic-scored on the untouched outer test. Each
   replica charges its own complete model. These replicas are scientific
   evidence, not one deployable packet.
2. **One final whole-panel two-part packet.** Nested fold votes choose one cell
   before full-panel fitting. Its Q0.16 values are then fitted, serialized, and
   followed by canonical terminated arithmetic payloads. Every payload is
   decoded and canonically re-encoded. Only this packet's literal byte ledger
   can hard-pass or hard-kill the physical target.
3. **Eight matched Gaussian controls.** Controls are inaccessible until the
   physical packet and pooled nested holdout both survive. All eight current
   Gaussian baselines must independently authenticate and replay before the
   first control fit. The source-selected cell is independently refit and
   charged on each. Controls can reject specificity; they can never turn a
   source physical failure into a pass.

A positive promotion status therefore depends on the exact physical packet,
sealed nested holdout, and all controls. In-sample fit alone never promotes.
Continuous NLL is not used as a hard-kill bound; exact integer arithmetic bits,
termination, byte padding, model bytes, headers, directory, immutable state,
alignment, the 2.15-bpw floor, and cold pages are charged.

## Long-memory source-free fixture

`fixture_long_memory.py` generates a frozen synthetic process whose symbol
probability depends on cumulative prefix parity since a 4096-symbol reset. A
two-state XOR unifilar cell knows the parity, while a suffix of depth at most
six does not. The fixture fits on independent generated streams and requires a
strict heldout advantage under the real canonical arithmetic coder. This proves
the v1 bank is not merely a renamed suffix table. It is not Qwen evidence.

## CPU and CuPy agreement

`uwfa_common.py` is a standard-library reference containing exact topology
updates, integer fitting, model serialization, the canonical 32-bit arithmetic
encoder, a causal decoder, and mandatory canonical re-encoding.

`cupy_backend.py` contains only CuPy RawKernels. One CUDA thread walks one
stream, with uint64 exact counts and exact arithmetic logical-length replay.
The authenticated producer performs an 8192-symbol multi-topology CPU/CuPy
equality preflight before any panel fit. CuPy is imported only after external
source-manifest/review authentication and exact replay of every source baseline.

## Fail-closed lifecycle

- A wrong token rejects before output, input, project import, CuPy, or CUDA.
- A valid token first reserves a new absent output directory.
- Using only the standard library, the runner rejects symlink leaves, opens the
  external independent review, authenticates `SOURCE_MANIFEST.json`, and hashes
  every declared member before importing same-package code.
- Source locks and every current arithmetic payload replay before CuPy import.
- `COMPLETE.json` is exclusive and last. Incomplete outputs are never resumed.

After an independent source auditor supplies a receipt bound to this exact
manifest, the only authorized launch form is:

```bash
/usr/bin/python3.12 -B -I \
  research/unifilar_wfa_entropy_census_stage0_v1/stage0_census.py \
  --authorization OPEN_AUTHENTICATED_UNIFILAR_WFA_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V1 \
  --review-receipt /absolute/path/independent_review.json \
  --stream-lock /absolute/path/current_stream_lock.json \
  --gaussian-control-lock /absolute/path/gaussian_control_panel_lock.json \
  --output /absolute/path/new-absent-output-directory
```

Do not run this command against any payload until the independent receipt's
`reviewed_source_manifest_sha256` equals the literal SHA-256 of this package's
`SOURCE_MANIFEST.json`.

## Universal claim boundary

The topology, fitting, nested split, selection, reset, control seeds, arithmetic
format, and decision criteria are model-agnostic. Qwen may be an evaluation
panel, never an architecture dependency or decoder key. Checkpoint-adaptive
Q0.16 values are permitted only because their bytes are serialized and charged.

A miss closes this frozen selected-SC-bit cell. It does **not** close a
source-coordinate label copula, arbitrary MPS, MERA transform, RCC backend, or
all universal SwiGLU-MoE codecs.
