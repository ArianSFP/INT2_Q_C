# Final independent source audit: UWFA-SC posterior-centroid v0

Date: 2026-09-02

## Verdict

**PASS**, strictly for the frozen source-only, nonpromoting diagnostic scope.

Producer identity:

- `SOURCE_MANIFEST.json` SHA-256:
  `0ef30253d4d31504fbd8f88b8203cf35bce6c14952e570aace44b7bc089cb713`
- source snapshot root:
  `ea3ad9cf9b723cdf7501eeff004bd7f2821af4d37ff186b72f2972482a05e11c`
- six source members: all byte counts and SHA-256 values independently
  recomputed and matched.

All earlier audit products and hashes were ignored. This audit compiled the
six frozen members from the newly authenticated retained byte closure.

This PASS does **not** authorize a positive Qwen, Gaussian-control,
portability, inference-read, or universal SwiGLU-MoE claim.

## What passed

### Source and loader authentication

- A copied package authenticated at the frozen manifest hash.
- Mutating one copied member after authentication made a fresh closure check
  fail.
- The already retained authenticated member bytes still compiled to the
  frozen digest after that live copied sibling was mutated.
- `posterior_core.py` and `result_bridge.py` are compiled from the retained
  owned closure rather than reopened from mutable siblings.
- Exact v8 decoder sources loaded under hash-bound private module names.
  Dataclass-bearing modules were registered in `sys.modules` before execution;
  the repaired loader path works.

### Leakage and cross-fitting

The independent synthetic call trace held outer ownership component 0 out and
observed every call to `fit_head`:

- no fit call contained owner 0;
- the two inner directions were exactly `1 -> 2` and `2 -> 1` for all eight
  frozen ridge exponents;
- the outer-fold refit used exactly components 1 and 2;
- source inspection confirmed that all three law heads are serialized and
  parsed through binary16 before `heldout_source` is opened;
- heldout scoring uses the parsed binary16 head, not retained FP64 parameters;
- owner components are derived from the complete stream-owner hypergraph.

The compressed UWFA state and lattice-index features are decoder-visible
message functions. Their use is not score-target leakage. The original BF16
values remain fit/score targets and are not serialized decoder inputs.

### Serialization and physical gates

- The maximum state-aware head is exactly 1,636 bytes: a 96-byte header plus
  770 little-endian binary16 parameters.
- Head and `CAGEPST1` wrapper canonical re-encoding passed.
- Binding, padding and wrapper tampering were rejected.
- The wrapper adds exactly one 4-KiB physical suffix page.
- A declared second compressed-expert invocation was rejected.
- Repeated byte requests that push the strict ledger above 2x were rejected.
- The rate gate is inclusive on `[2.15, 2.5]`, the target gate is `F <= 0.8`,
  and cold read is strict `< 2x`.

### Completion-last publication

The producer uses POSIX directory descriptors. The Windows audit host refuses
an `O_RDONLY` directory open, so the unmodified publication test cannot execute
that one primitive here. An audit-only compatibility proxy preserved real
exclusive member-file descriptors, real writes and file fsyncs while replacing
only the unavailable directory descriptor/fsync. The resulting event trace
proved:

1. all ordinary members were opened and fsynced first;
2. the directory durability barrier occurred;
3. `COMPLETE.json` was the last member opened and fsynced;
4. a second directory durability barrier followed completion; and
5. an existing result directory was rejected.

No producer byte was changed.

### Exact predecessor scope

The executable does **not** require a standalone v9 source-model survivor.
`authenticate_result_directory` requires the exact completed v9 schemas,
members, hashes, a literal `UWFCV8` container, and nonpromoting boundaries; it
does not gate on a survivor status. A synthetic completed v9 record explicitly
marked `HARD_KILL` authenticated and exposed its literal container for joint
scoring. Mutating that container was rejected.

Therefore posterior v0 can honestly test a joint rate-plus-centroid pass even
when the v9 WFA saving alone is insufficient. The design-lock phrase
`hard_fail_without_literal_survivor_container` is ambiguous and should be read
as “hard fail without an authenticated completed literal container,” or
clarified in a later source revision.

## Explicit inference-read boundary

The routed proof actually executes the authenticated v8 inner routed decoder
once per expert and reads/parses the suffix page. It deliberately does **not**
accumulate posterior occupancies and apply the posterior head inside that same
routed session. The source, ledger, README and prospective result all agree:

- `actual_posterior_wrapper_routed_decode_executed = false`;
- `posterior_head_applied_to_routed_reconstruction = false`;
- no compressed second pass is allowed; and
- the reported cold-read value is a nonpromoting projection only.

This is an acceptable boundary for a discovery diagnostic. It is not an
inference-read proof. Promotion requires an inference-ready routed posterior
decoder that reproduces the offline output without rereading the inner expert.

## Universality boundary

The model head uses only decoded lattice indices, decoded pre-decision state
occupancies, public shape/role information and serialized parameters. The
source manifest grammar rejects checkpoint, layer, tensor-name and model-family
identity fields. The design explicitly forbids identity lookup and external
reference weights.

Nevertheless, the current implementation is bound to the existing v9/STRATA
adapter and is a Qwen-panel discovery diagnostic. It contains no portability
evidence and must not be described as a universal SwiGLU-MoE performance
result. Algorithmic universality remains a future sealed-family test.

This diagnostic also does not close the full TACTIC-CAGE proposal. It tests one
decoder-legal, block-occupancy affine posterior on a completed `UWFCV8`
container. Coarse-code-programmed graphs, lifting, adaptive 384-bit refinement
trees and syndrome reconstruction remain separate hypotheses.

## Test receipt

The producer’s exact unmodified source test ran 22 tests on the Windows audit
host:

- 20 passed;
- 0 assertion failures;
- 1 error: Windows rejected the POSIX directory descriptor in
  `_write_exclusive`;
- 1 skip: symlink creation unavailable; and
- no payload or CUDA access was reported or observed.

The independent clean-room audit then passed manifest mutation, retained-byte
loading, exact v8 dataclass loading, cross-fit exclusion, canonical grammar,
tamper, second-pass, over-2x, completion-order and completed-hard-kill input
checks. Its process exited 0.

The README example uses relative paths for the v8 package and two decoder
sources, while the authenticated readers require absolute paths. This is a
nonblocking documentation defect: the actual launch must provide absolute
paths.

## Access attestation

This audit did not open, stat, hash, enumerate or numerically materialize:

- any Qwen/model payload;
- any completed v9 result;
- any BF16 score source;
- any Gaussian control; or
- any RunPod/CUDA object.

Only frozen source files, source manifests, temporary synthetic byte objects
and temporary synthetic BF16 fixtures used by the producer’s own source tests
were accessed.
