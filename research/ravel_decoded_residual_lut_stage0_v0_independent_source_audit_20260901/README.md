# Independent source audit: RAVEL-6144-v0

Verdict: **BLOCK — do not launch v0.**

This is an independent, source-only audit of the exact seven-file producer
package `research/ravel_decoded_residual_lut_stage0_v0`. The audit opened no
model payload, decoded panel, fresh validation data, CuPy runtime, GPU, or
network resource and did not modify the producer.

## Fatal projection defect

The legal correction applied by the runner is

```text
corrected_error_i = residual_i - table[cell_i] * row_rms_i.
```

For one lookup cell, the raw-SSE minimizing entry is therefore

```text
sum(row_rms_i * residual_i) / sum(row_rms_i**2).
```

The runner instead averages `residual_i / row_rms_i` with an unweighted
coordinate count. That minimizes normalized error, not the declared raw MSE.
Row RMS is not constant within a cell, because the key contains only one of
four RMS classes. The claimed holdout-self-fit oracle consequently need not
dominate every legal shared RAVEL table.

The verifier carries a two-coordinate counterexample. With scales `(1, 2)`
and residuals `(1, 1)`, v0 emits `t=0.75` with raw SSE `0.3125`; the legal
least-squares entry is `t=0.6` with raw SSE `0.2`. A v0 `oracle_F > 0.8`
could therefore reject an architecture whose actual favorable projection
survives. This invalidates the hard-kill rule before payload launch.

## Other findings

- **External-use binding: BLOCK.** The plan, header, decoded reconstruction,
  and each BF16 source are hashed and then reopened by pathname for parsing or
  mapping. The bytes actually used are not the held authenticated bytes. The
  same issue affects the retrospective script hash in the result.
- **Feature visibility: PASS with documentation defects.** Role, row RMS,
  amplitude, and neighbor states depend only on decoded reconstruction and
  fixed metadata. However `cp.roll` makes horizontal edges cyclic: column zero
  uses the final column as its left neighbor, and vice versa. The README does
  not disclose this. Exact RMS thresholds, amplitude boundaries, zero-sign
  convention, and magnitude-tie convention also need to be part of the codec
  specification.
- **Split/leakage: PASS.** The finite FP16 table is accumulated only from fit
  experts `{0,2,3,5}` and evaluated only on holdouts `{1,4}`. The second table
  intentionally uses holdout residuals and is correctly labeled source-leaky.
- **Favorable transfer: PASS as conditional arithmetic.** Multiplying the
  measured correction ratio by `2**(2*side_bpw)` is the preregistered favorable
  allowance for reducing the coarse payload by the side rate. The source
  correctly says survival is permission for controls/re-encoding, not target
  evidence.
- **Rate/read ledger: PASS.** The 16,384-byte packet costs exactly
  `0.004629629629629629` bpw on 28,311,552 weights, leaving
  `2.4953703703703702` bpw for the coarse payload. Four added 4-KiB pages add
  `0.011111111111111112` to the published `1.1694444444444445` worst read,
  yielding `1.1805555555555556 < 2`.
- **Authorization/output ordering: PASS with a partial-output limitation.**
  Authorization, device selection, output absence, and plan resolution all
  precede CuPy import and payload access. The output directory uses exclusive
  creation and each file uses `O_EXCL`; a crash after the packet write can
  nevertheless leave an intentionally recognizable partial result directory.
- **Baseline bindings: arithmetically consistent but not safely held.** The
  plan/header/decoded hashes and baseline constants agree between the design
  and runner; `500.39553685426534 / 16192.89450885593 * 32` is exactly
  `0.9888693569009007`. The hash-then-reopen defect prevents a strong claim
  that authenticated bytes are the bytes used.
- **CuPy path: structurally present.** Feature construction, cell accumulation,
  evaluation, and sums use CuPy with FP64 accumulation. No runtime/GPU claim is
  made by this source audit.
- **Packet: parseable in principle, underspecified and unchecked.** Canonical
  JSON plus LF is self-delimiting, followed by 12,288 FP16 bytes and zero
  padding to 16,384 bytes. The current header is 79 bytes, so the FP16 payload
  begins at an odd offset and leaves 4,017 padding bytes. No parser,
  round-trip test, alignment rule, padding validator, or FP16 finiteness check
  is supplied. Casting a large learned value to FP16 could silently serialize
  infinity.

## Minimum v1 repair contract

1. Compute the source-leaking raw-MSE oracle with per-cell numerator
   `sum(scale * residual)` and denominator `sum(scale**2)`. Independently test
   it against a direct scalar least-squares solve and assert it never loses to
   any compared legal table, including the emitted fit table.
2. Authenticate external inputs through held regular-file descriptors (or an
   equivalent immutable authenticated snapshot), then parse/map the same
   bytes. Reject links and identity changes; do not hash one pathname open and
   consume a later one.
3. Freeze and document boundary behavior and every bin/tie convention. If
   cyclic neighbors are intended, state that explicitly; otherwise add a
   decoded-visible edge sentinel or clamp rule and adjust geometry.
4. Define a fixed, aligned packet header with an explicit header length,
   payload length, version, endianness, and padding rule. Supply an independent
   parser/round-trip test, reject nonzero padding, and reject nonfinite FP16
   entries before creating output.
5. Preserve the current whole-expert split, favorable-transfer limitation,
   side-rate/read ledger, authorization-before-access order, atomic directory
   collision check, source/output hashes, and narrow claim boundary.

## Replay

On a machine with a native, non-link Python 3.12 executable:

```bash
INT2_PROJECT_ROOT=/workspace/INT2__compression/INT2_Q_C
/usr/bin/python3.12 -B -I \
  "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v0_independent_source_audit_20260901/verify_audit.py" \
  --audit-dir "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v0_independent_source_audit_20260901"
cd "$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v0_independent_source_audit_20260901"
/usr/bin/python3.12 -B -I test_audit.py
```

These commands use only the sealed audit and its copied source evidence.
