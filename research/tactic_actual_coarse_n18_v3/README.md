# TACTIC actual-coarse N18 v3 — source-only closure candidate

This sibling package repairs the security and universality blockers found by the independent v2 audit. It is deliberately not a codec result. It has not read Qwen or any model payload, run a numerical producer, imported the pinned CuPy codec dependencies, frozen a runtime, executed CUDA, emitted results, created a source manifest, transferred to RunPod, or been committed.

## Frozen mathematical boundary

For every complete 4,096-weight microblock, the coarse slot is exactly 1,228 bytes, or `307/128 = 2.3984375` logical bpw. Sixty-four such slots form the 78,592-byte N18 reservoir. The frozen DH384 handoff adds 48 fine bytes and four metadata bytes per complete microblock, giving 1,280 bytes or exactly 2.5 bpw.

The shape API accepts arbitrary positive SwiGLU expert dimensions within explicit integer/product caps and treats Gate, Up and transposed Down as three equal-sized roles. It never allocates a full N18 reservoir to an incomplete group. Role tails are combined into one canonical aggregate tail. Experts whose entire 2.5-bpw byte budget is below 8,191 bytes use an implicit all-zero reconstruction with no packet and no routed read. Larger experts receive exactly `floor(5 * weights / 16)` contiguous bytes. This fallback closes rate and I/O for tiny experts; it makes no quality claim.

Routed traffic is charged as the union of distinct 4 KiB pages intersecting an owner's contiguous frame, not as nominal bytes. Every explicit frame is large enough that this union is strictly below twice its stored bytes, for every starting-page offset. Implicit frames read zero pages. The reference 768×2048 Qwen geometry remains exactly 2.5 bpw and one page-byte per stored byte; the hostile 769×2051 geometry and 1×1 geometry are explicit test cases.

The frozen best-of-64 DH384 pilot has a narrow interpretation: failure can kill that DH384 branch only. It cannot kill coarse-programmed graphs, adaptive lifting, posterior reconstruction, syndrome refinement, joint coarse/fine search, or broader TACTIC-CAGE mechanisms.

## Authenticated execution boundary

There is no supported live-tree entry point. An independent dispatcher must:

1. open and externally pin `immutable_bootstrap.py`;
2. inherit that held descriptor into an isolated Python process;
3. execute `/proc/self/fd/N` and pass the same FD plus its externally pinned SHA-256;
4. supply a separately held, externally pinned flat inventory and expected source root;
5. keep the source directory an exact flat closure with no `__pycache__`, symlinks, directories or extra files.

The bootstrap reads every bounded source member through held no-follow descriptors and creates immutable byte packets. A dedicated loader compiles those exact packets, assigns pseudo-paths, disables bytecode, removes the live package/current directory from `sys.path`, rejects preloaded siblings, and checks that every loaded sibling used its authenticated-byte loader. The bootstrap FD is rechecked on exit. The bootstrap itself does not create trust; the external procfd launch is the trust boundary.

Pinned prototype dependencies are not imported by this package. A future dispatcher must first authenticate their held bytes and exact AST import closure, then authenticate a real runtime lock. That lock covers held interpreter bytes and exact `sys.version`, and for CuPy, NumPy, pynvml and SciPy covers version, RECORD bytes/hash, and a domain-separated tree hash of every `importlib.metadata` file. Empty, zero, negative and all-zero placeholder fields fail closed.

## Publication and authority

The future publisher writes into a private sibling staging directory. Ordinary artifacts are written and synced first, then `ARTIFACTS.json`, then `COMPLETE.json`. It syncs, exactly enumerates and rehashes the complete staging tree, restricts and re-syncs it, and only then performs a single `renameat2(RENAME_NOREPLACE)`. Before that rename, injected failures clean private staging and cannot expose a public COMPLETE tree. Failures after rename can leave only the already verified tree. Constructor phases and cleanup are explicit.

This package cannot issue a review seal or authorize a payload run. It only consumes an opaque assertion FD already authenticated by an independent dispatcher and bound to source root, runtime root, action and external audit evidence.

`safe_telemetry.py` is only a strict receipt validator for a future instrumented CuPy run. It requires CUDA/NVML UUID and PCI agreement, nonempty runtime identities, per-buffer transfer provenance and digests, synchronized CUDA event phases, wall time, and sampled memory baseline/peak/delta with an explicit sampling caveat. It is not evidence that such a run occurred.

## Files

- `v3_common.py`: strict primitives, caps and frozen constants.
- `universal_layout.py`: universal role/tail/fallback partition and unique-page ledger.
- `immutable_bootstrap.py`: external-inventory authentication and immutable byte execution.
- `dependency_auth.py`: safe dependency graph and immutable dependency packets.
- `runtime_auth.py`: held executable plus distribution RECORD/tree authentication.
- `secure_io.py`: held inputs and verify-before-publication state machine.
- `dispatcher_contract.py`: external assertion consumer; no issuing authority.
- `safe_telemetry.py`: future CuPy/NVML receipt validator.
- `verify_source.py`: authenticated source-only invariant verifier.
- `test_source_only.py`: hostile source tests, including POSIX publication fault tests.
- `design_lock.json` and `dependency_graph.json`: declarative contracts.
- `POSTIMPLEMENTATION_REVIEW.md`: stopped producer review and limitations.

## Audit handoff

The independent auditor must create and own any external inventory, source-root pin, runtime lock, review conclusion, or execution authorization. The Windows implementation host can run arithmetic, schema, immutable-loader and static tests, but cannot execute POSIX no-follow/procfd/`renameat2` fault tests. Those tests are authored and must be replayed independently on POSIX before any payload authority is considered.
