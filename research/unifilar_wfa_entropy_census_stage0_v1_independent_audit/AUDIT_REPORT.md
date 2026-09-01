# UWFA census v1 independent adjudication

## Result

`BLOCK_INDEPENDENT_SOURCE_REVIEW`. Payload authority is not granted.

The finite WFA mathematics passed: all 150 cells matched CPU and CuPy exactly,
75,600 independent transition cases passed, 2,046 exhaustive small arithmetic
cases matched an independent encoder, serialized genuine models round-tripped,
resets occur before the `t=0` emission, the nine-fold synthetic nested protocol
replayed deterministically, and relabelling layer/expert/stream identity did not
change probability-model inputs.

The block is caused by six evidence-binding defects, recorded with executable
synthetic demonstrations in `HOSTILE_TEST_RESULT.json`:

1. Source members are closed after authentication and reopened by pathname for
   import, permitting replacement between review and execution.
2. Symlink leaves reject, but symlinked ancestors are accepted and unpinned.
3. `current_object_bytes` is not bound to the authenticated current-artifact
   byte count, so the physical-saving denominator can be inflated.
4. Final acceptance decodes with in-memory parameters, not the emitted model;
   an all-`0xFF` serialized model was accepted in the synthetic hostile test.
5. The claimed full physical packet is not emitted. Headers, directory,
   immutable state, padding, and page placement exist only in a ledger.
6. Eight control baselines do replay before fitting, but the runner never
   enforces matched source/control geometry.

The measured CuPy count-kernel times at 131,072, 262,144, and 524,288 symbols
were 0.001326, 0.002585, and 0.005102 seconds. This supports linear fixed-cell
work. The complete nested search remains
`O(outer_folds * 150 * N)`.

This adjudication is deliberately narrow. It neither launches a payload nor
closes arbitrary MPS/MERA/source-coordinate copula approaches.
