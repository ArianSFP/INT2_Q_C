# Independent bridge audit, 2026-09-02

Verdict:

```text
PASS_FOR_NONPROMOTING_REAL_QWEN_EARLY_KILL_RERUN
BLOCK_FOR_ANY_POSITIVE_OR_PRODUCTION_CLAIM
```

The independent reviewer authenticated:

```text
early_gate.py
  399cb25260d34ec299cc91a17f129da9be5ba5b799c961e43f0c1b0637ee0174
test_source_only.py
  306c5d4a822b2300c266eb250a26b0eb8543865b4c5a6684edfd2528ad62afbd
design_lock.json
  df420497a2c7deb4d7e3bfc9ddd7632b5981f39630394829d7d1f42ba1374fed
sealed v8 manifest
  a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6
sealed adapter
  08fc8808ac168f6930ee9482e160f25f2bd087829fca4630553aea3510d722c6
STRATA common source
  3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1
frozen external auditor
  85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6
```

The RunPod source-only suite passed 12/12.  A targeted test against the real
pinned STRATA helper covered all 15 returned rows and all 13,824 ordinals.  It
proved that the bridge produces exact `list[list[int]]`, preserves every value
and its order, and gives the same NumPy selection as the original one-
dimensional `np.ndarray` row.  All three ordinal uses in the sealed adapter are
compatible with the list representation.

The active `20260902f` command was verified to use this exact runner, `-I -B`,
and the pinned external paths.  It may produce a nonpromoting early-kill
diagnostic only.  The wrapper bridge and single-artifact cache are exploratory,
matched controls are absent, the wrapper is not sealed producer authority, and
the environment is not the production dispatcher closure.  A source survivor
must be rerun through those later gates before any positive claim.

After the active process had authenticated its clean input snapshots, an
auxiliary audit invocation without `-B` created an undeclared `__pycache__` in
the isolated checkout.  This did not alter the held source inputs of the active
process.  The checkout is nevertheless retired after `20260902f` and must not
be reused for another candidate run.
