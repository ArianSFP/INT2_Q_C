# UWFA-SC v8 external dispatcher v2 independent source audit

Date: 2026-09-02

Verdict:

```text
BLOCK_PRODUCTION_AND_PAYLOAD_EXECUTION
```

The candidate is a substantial improvement over v1, and its fresh RunPod
POSIX suite passed 40/40.  That suite is not sufficient: independent review
found one fatal real-producer ABI mismatch and one root-of-trust flaw.

No Qwen, model, control or candidate-result payload was accessed during this
audit.  V1 remains untouched.

## Authenticated candidate

```text
SOURCE_MANIFEST.json
fac1c4b2b66ae968aa081676924f3a64962f54c07e401fd0e4d4db978d238bcb

README.md
546dda01f8ba54d246cd7b3030592e544fbdf4f98a481a8595cd67c1e0a48bd3
bootstrap.py
c2f849eadb2e4edc9d50f9fd456fb8cbe741e6e24fdbee7b5165d1fa5a742e2f
decoder_bundle.json
fc0116b46b4d670e03231e1c5eb89456b5db955c00353d256c6ae57b513b33db
design_lock.json
b0ec9aa932940b0e7e71562bb51d9fab9fba64a8ef451aefc89c50a9dc24ffea
runtime_lock.json
6e533061aa4a531530760d39fd92a6e464db21b6030bade2aeb62f0a0e38738d
strata_ordinal_bridge.py
a01b443cbae74c7a7cb0768fe186046de5d65351b5a6fae279fb32abeb7da99d
test_source_only.py
92bc70ffe4916b88b8ab770fe583ef36023bcb2770ebcead9c91ecffafa4c0d1
verify_output.py
78f967694099803c30fe95af66975e87e615cd3f5fdb403d7278f3a54534520f
verify_source.py
0a8f5108f8ad6466e79921d61374bd4023bdc9b8831a4cab85c2ad5524195027
```

The pinned producer source is:

```text
strata_expert_local_codec/common.py
3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1
```

## Blocker 1: real ordinal-row ABI rejects before conversion

The pinned helper's `expected_block_group_ordinals` returns an outer built-in
`list` whose rows are one-dimensional NumPy arrays.  Dispatcher v2 requires
every raw row to have exact built-in `list` type before its scalar bridge runs.
Consequently every real Qwen row is rejected, even though the output contract
correctly requires `list[list[int]]`.

The green test constructs fabricated list rows and therefore does not exercise
the pinned producer ABI.

The repair must:

1. require the exact outer built-in list;
2. require every input row to be an exact one-dimensional NumPy array with an
   integral dtype and the expected partition geometry;
3. convert each NumPy scalar through the existing exact integer round-trip;
4. emit exact built-in lists containing exact built-in integers;
5. preserve order, cardinality, uniqueness and complete group coverage;
6. prove `post[converted_row]` equals `post[original_ndarray_row]` on the real
   pinned helper, while tuple and fabricated list producer rows reject.

## Blocker 2: retroactive module provenance is forgeable

`_bind_preexisting_interpreter_modules` attempts to authenticate a preloaded
module object by inspecting mutable presentation metadata:

- a file-backed object can claim a `__file__` path that belongs to an unrelated
  held source member;
- an object without a file/spec can claim an absent or `None` origin and be
  labelled interpreter-intrinsic.

Neither check proves that the object executed the held bytes.  The later hash
chain faithfully records the incorrect initial binding, so append-only logging
does not cure the root-of-trust error.

A sound successor must never retroactively authenticate a file-backed module
from `__file__`, `__spec__`, loader or origin metadata.  It needs a tiny held
stage-0 prelude, installed before every non-frozen import, that registers
module-object provenance at authenticated execution time.  The late dispatcher
may then accept only:

- objects already bound by that authenticated loader/registry;
- genuine built-ins whose name, origin and exact built-in importer agree with
  the interpreter;
- genuine frozen modules whose origin and exact frozen importer agree.

Origin `None`, preloaded source/extension modules and later path claims must
reject.  A source-only test must cover a foreign module object claiming an
actually held benign path, a no-file/no-spec object, a forged loader/spec, and
import-then-remove history.

## What v2 did close

Subject to the blockers above, v2 adds useful machinery worth preserving:

- held-byte compilation for post-enforcement modules;
- locked import hooks and empty ambient search path;
- an append-only hash-chained Python/native-load event ledger;
- process-start native-auditor feed requirements;
- transient load/unload retention;
- strict repeated-read and owner-bandwidth checks;
- exact output/result revalidation;
- the correct *output* ordinal ABI (`list[list[int]]`).

These are implementation assets, not payload authority.

## Promotion boundary

Do not seal, launch on a payload, or use v2 as evidence for a positive or
negative Qwen result.  Build a new sibling with the exact ndarray bridge and
an independently audited process-start stage-0 provenance root.  Rerun all
source-only, POSIX hostility, real-helper ABI and native-event tests before
resolving any payload path.
