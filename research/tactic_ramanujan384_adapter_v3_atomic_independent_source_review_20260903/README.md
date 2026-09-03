# Independent source review: TACTIC Ramanujan-384 atomic adapter v3

Date: 2026-09-03

Pinned producer inputs:

- v3 manifest SHA-256: `97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1`
- v3 source root: `5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674`
- external bootstrap SHA-256: `f7e8cd469b0ff9dd9ef09b400c63ec9f91e067f849d6b009588ea94ad6494375`

This is a benign source-only review. No network, Qwen/model/coarse payload,
NumPy/CuPy fixture, or production decoder was opened or executed. The producer,
v2, v2 review, and external bootstrap were not modified.

## Result

The requested v2 repairs are materially implemented.

- The supplied v3 manifest is an exact eleven-member flat closure and its
  canonical member-row root independently recomputes to the supplied root.
  The external bootstrap and the pinned v2/v2-review manifests also match their
  dependency-lock hashes and source roots.
- Before any v3/v2 package source is compiled, the bootstrap descriptor-reads
  every pinned member, requires a regular single-link object, checks exact
  closure both before and after, copies only authenticated bytes to a private
  read-only tree, re-reads and rehashes it, and returns an immutable
  `MappingProxyType[str, bytes]`.
- `snapshot_runner.py` and all used v3/v2 project modules are compiled directly
  from immutable mapping values. There is no filesystem module loader in the
  runner and no import from the mutable producer trees.
- `run_authenticated_expert` accepts no decoder argument. The coarse program
  is canonical JSON data interpreted by a fixed authenticated Python function;
  it receives literal coarse bytes, uint32 geometry, and the fixed role tuple.
  Its exact schema permits no path or callback field and requires zero imports.
- Worker source, auditor source, capability, and receipt have separately named
  external hash parameters. Both source closures and the receipt schema are
  authenticated, the program is bound to the coarse hash/shape/roles, and
  decoded role bytes must match the receipt's independently recorded hashes.
- The pinned v2 core remains byte-identical. V3 retains its one batched solve,
  batched all-rank reconstruction, zero per-candidate host scalar transfers,
  exact literal composite replay, physical `2.15 <= R <= 2.5`, absolute
  `D <= 0.025`, one phase plus eight canonical Gaussian full-search controls,
  and `>=0.03 bpw` source-minus-strongest-control promotion gate. The frozen
  `[128,2048]` fixture ledger is 786,432 weights and 245,760 bytes, exactly
  2.5 bpw.

## Findings and conditions

### A1 — the package snapshot boundary passes

Once the external bootstrap is trusted, package mutation after authentication
cannot affect compiled project code: all compilation uses the verified bytes
held by the mapping proxy. This closes v2's verifier-to-import race.

### A2 — bootstrap authenticity still requires a pre-execution launcher

Python necessarily reads and begins executing the external bootstrap before
line 203 self-reads and hashes `__file__`. A transiently substituted bootstrap
could execute and restore the pinned bytes before that self-check. The script
also does not require isolated mode (`-I`) or reject `PYTHONPATH`; its standard-
library imports occur before its self-hash and may be shadowed from the script
directory in an untrusted launch environment.

Therefore the supplied bootstrap digest is a valid publication pin, but the
script cannot establish its own pre-execution authenticity. A trusted launcher
must hash/open the bootstrap first and execute those authenticated bytes in an
isolated interpreter, or the bootstrap must be part of an immutable host image.

### A3 — worker/auditor independence is not source-enforced

`authenticate_and_decode` labels six expected values “distinct external
SHA256 pins” but checks only that each is a 64-character string. It does not
require `len(set(pins)) == 6`, distinct worker/auditor roots, distinct physical
directories, or a trusted signer. Because the source root hashes only member
rows, worker and auditor manifests with different schemas can intentionally
refer to identical member bytes and the same root. All expected hashes are
caller inputs.

This is sound only when a genuinely independent party publishes the pins and
the launcher trusts that publication. Self-generated source closures and PASS
receipts demonstrate schema mechanics, not independent authority. Require
non-aliased roots/directories and bind the accepted pin tuple in an external
signed launch manifest before production.

### A4 — the worker VM is safe but fixture-only

The frozen VM recognizes only `ZERO_F32_LE`; it cannot decode a real nonzero
coarse codec. That is consistent with the package's explicit mechanism-only
status. A future production program/opcode and an independently controlled
audit are still required, as are practical allocation limits below the current
uint32 geometry ceiling.

### A5 — runtime remains pending

The producer correctly records that Python tests, atomic verify-only replay,
CPU/CuPy fixtures, model payloads, and coarse payloads were not executed here.
No compression or portability claim follows from this source review.

## Disposition

```text
PASS_V2_ATOMIC_SNAPSHOT_BYTE_WORKER_AND_SCIENTIFIC_SEMANTICS_REPAIRS__
CONDITIONAL_ON_PREVERIFIED_ISOLATED_BOOTSTRAP_AND_EXTERNALLY_INDEPENDENT_NONALIASED_WORKER_AUDIT__
HOLD_REAL_COARSE_DECODER_PYTHON_CUPY_PAYLOAD_AND_RD
```

## Replay

From this review directory:

```powershell
.\review_static.ps1 `
  -Producer ..\tactic_ramanujan384_adapter_v3_atomic `
  -Bootstrap ..\tactic_ramanujan384_adapter_v3_atomic_bootstrap.py
```

```bash
python -I -B run_review.py \
  --producer ../tactic_ramanujan384_adapter_v3_atomic \
  --bootstrap ../tactic_ramanujan384_adapter_v3_atomic_bootstrap.py \
  --output /tmp/tactic-r384-v3-independent-source-review.json
```
