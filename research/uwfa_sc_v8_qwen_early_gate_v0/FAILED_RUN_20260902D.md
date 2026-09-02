# Qwen early-gate run 20260902d: implementation failure

This run is not a scientific Qwen result.

The detached RunPod process completed the single causal decode of the pinned
Qwen STRATA artifact, then exited before creating a result directory:

```text
FAIL_UWFA_SC_V8_QWEN_EARLY_GATE: IndexError: too many indices for array
```

The exploratory ABI bridge had converted each one-dimensional NumPy group-
ordinal array to a `tuple[int, ...]`.  That satisfied the sealed adapter's
exact-scalar-type check, but it did not preserve container semantics.  The
adapter later evaluates `post[group_ordinals]`; NumPy interprets a tuple as one
index per axis, so any row longer than the array rank raises the observed
exception.  A Python list of native integers preserves the original one-axis
advanced-index operation.

The repaired bridge returns `list[list[int]]`.  Its source-only regression test
now proves both exact native-integer conversion and equality of the resulting
NumPy selection to selection by the original one-dimensional NumPy array.  It
also proves that the rejected tuple representation triggers the observed
failure.

No entropy saving, Qwen-minus-control advantage, rate, distortion improvement,
or promotion status may be inferred from run 20260902d.  Its log and exit
receipt remain preserved under the `20260902d` RunPod namespace; the repaired
run uses a new namespace and isolated checkout.
