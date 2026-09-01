# Independent hostile source audit: label-copula stage 0 v0

## Verdict

**BLOCK_INDEPENDENT_SOURCE_REVIEW** for source manifest
`e1bc2873f204b1db5fefa666d0daf6ddebae38bd3f1add3ce45f3bc0538aae14`.

No Qwen/model tensor, current-codec artifact, Gaussian-control payload, network
resource, CuPy module, CUDA context, or GPU job was opened. The producer package
was not modified.

## Independently reproduced passes

- The external manifest hash matches and closes over exactly the declared
  source members.
- All 18 native source-only tests pass. The synthetic parity fixture passes,
  including real arithmetic round trips and a suffix-collision/nonlocal-state
  witness.
- An independent semantic reference reproduced arbitrary non-square and
  singleton `Gate[j,k], Up[j,k], Down[k,j]` traversal.
- The fixed RMS/Gaussian-Lloyd4 decision boundaries are exact and do not fit
  parameters on validation or test streams.
- Independent formulas reproduced every state transition over all 240 cells,
  all states, both symbols, roles, planes, and reset-boundary positions.
- An independent decoder parsed serialized Q0.16 packets and decoded producer
  arithmetic payloads for every topology and both probability extremes.
- Layer/expert strings do not enter probability fitting. A regular grid has
  whole-layer outer and whole-slot inner isolation.
- Independent arithmetic closes the model/header/directory/frame/alignment/page
  storage ledger. The source-first promotion threshold is exactly
  `0.1528899669629145` physical bpw and controls cannot rescue a failed source
  lower bound.
- The claim boundary is correctly diagnostic raw labels only. Current STRATA
  transformed symbols remain deferred.

## Blocking defects

1. `LIFECYCLE_COMPLETE_NOT_FINAL`: `CompletionLastOutput.write_new` remains
   callable after `complete`; the hostile probe successfully created
   `AFTER_COMPLETE.json` after `COMPLETE.json`. The helper therefore does not
   enforce the documented exclusive-last invariant.

2. `SOURCE_PACKAGE_SYMLINK_ACCEPTED`: `verify_package` resolves the package
   argument before checking `is_symlink`, erasing the link identity. A
   symlinked package was accepted by the verifier.

3. `DEGENERATE_ONE_CLUSTER_CONFIDENCE_GATE`: `nested_partition` accepts three
   layers and chooses one outer test layer. All 4096 cluster-bootstrap replicas
   are then identical to the point estimate, yet the value is labeled a lower
   95% bound and is eligible to promote. Require a defensible minimum number of
   outer clusters or fail the uncertainty gate.

4. `GAUSSIAN_CONTROL_PROVENANCE_UNBOUND`:
   `evaluate_independent_matched_controls` accepts arbitrary prebuilt
   `SymbolStream` panels. It checks only that there are eight panels, then
   reports the frozen seeds; it does not bind seed order, source geometry,
   source block moments, or evidence that each panel passed through independent
   Gaussian generation, canonicalization, RMS normalization, Lloyd labeling,
   nested selection, and physical coding. Controls cannot create a source pass,
   but the claimed full-pipeline control result is unauthenticated.

5. `EXPERT_SLOT_UNIVERSE_NOT_VALIDATED`: the documented inner split is over
   reusable expert slots, but `nested_partition` accepts irregular panels with
   layer-unique expert IDs. It does not require one unique stream per
   layer/slot or a common slot universe, so the promised whole-slot holdout is
   adapter-dependent and unenforced.

## Nonblocking scope observations

Each fixed state recurrence is O(1), so each candidate scan is O(N) and the
frozen bank is O(240N). The scientific implementation is CPU reference code;
the only CuPy action in sealed v0 is a late availability import. That is honest
for a no-payload preflight, but it is not yet the promised CuPy-optimized
scientific runner.

The raw-label saving is a diagnostic source statistic, not an operational
2.15–2.5 bpw reconstruction gain. It must not be credited to the current STRATA
codec without one nested, independently decoded end-to-end reconstruction and
byte/read ledger.
