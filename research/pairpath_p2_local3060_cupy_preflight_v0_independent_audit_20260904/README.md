# Independent audit: PAIRPATH-P2 local RTX-3060 CuPy preflight v0

## Verdict

**Runtime PASS; scientific/payload authority remains BLOCKED.**

The frozen receipt was authenticated at SHA-256
`a6c1fd514ddafa5a3225a4c70b030cf80df75a41f127f829f8ccd4b92cbe53ab`.
Every source hash recorded by it matches the exact six-member target closure,
including the source-only HOLD gate. The target files were hashed before and
after this audit and did not change.

On the actual pinned local GPU:

- device: `NVIDIA GeForce RTX 3060`;
- UUID: `GPU-458a424a-76e3-65e5-0470-803e0ed131ca`;
- CUDA runtime/driver API: `12090` / `12060`;
- CuPy/NumPy: `14.2.0` / `2.5.2`;
- all six solver parity rows matched CPU exactly;
- both complete RD rows, both hulls, equal-rate/equal-MSE comparisons and the
  final convexified gate matched CPU exactly;
- the global Up/Down bit weight was independently checked as
  `0.017669214863729712` on the unequal-energy complete-oracle fixture.

The receipt's analytical memory ledgers reproduce exactly. A fresh 1/8
allocation exercise used `13,369,344` live allocator bytes after one joint
update and left a `30,539,776`-byte allocator pool high-water observation,
against `17,568,128` explicitly enumerated bytes. These are not contradictory:
the analytical ledger explicitly excludes allocator workspace/caching.

Fresh 24,576-coordinate update timings were `0.00267765 s` independent and
`0.00282114 s` joint, giving linear kernel-only projections of `19.00 s` for
1/64 and `152.03 s` for 1/8. The frozen receipt measured faster projections of
`9.57 s` and `76.59 s`. This variability confirms that the figures are useful
feasibility microbenchmarks, not end-to-end runtime evidence; they omit CPU
canonical scoring/copies, setup, controls, and full-aperture measurement.

## Why authority remains blocked

The CuPy backend faithfully accelerates the optimistic single-letter oracle,
including its defect. On the deterministic legal-level counterexample, CPU
and GPU return the same joint objective `11.976051614873764`, while a label
assignment already found by the independent solver is legal under the same
joint model and scores `11.901158026223808`. The reproduced suboptimality is
`0.07489358864995665`. Exact GPU parity therefore does not create the missing
dominance/global-optimality certificate required for a scientifically safe
hard kill.

The backend itself correctly uses one global Up/Down multiplier. It does not
exercise or repair the downstream r2 finite encoder's role-local multiplier
bug, nor the finite packet decoder's failure to replay/validate its tree
descriptor. Any payload capability inheriting r2 would inherit all three
scientific/packet blockers.

No Qwen/model payload, network, remote host, RunPod, or production execution
was accessed. `AUDIT_REPORT.json` records the exact measurements.
