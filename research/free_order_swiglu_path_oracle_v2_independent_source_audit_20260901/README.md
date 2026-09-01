# Independent hostile source audit: FOSP-ARX-v2-DIRECT

Verdict: **BLOCK**.  The exact frozen producer package is mechanically
reproducible, its direct gross 3x3 reuse relaxation really does contain every
legal exact-coefficient path, and its physical rate/read ledgers are correct.
It must nevertheless not be authorized or run on source data in this form.

## Blocking findings

### FOSP2-SCI-001 — subtracting a relaxed control destroys containment

For each panel separately, target-wise predecessor reuse is a valid optimistic
upper bound on that panel's legal-path capture.  It does **not** follow that

`s_qwen_relaxed - mean(s_control_relaxed)`

upper-bounds

`s_qwen_legal - mean(s_control_legal)`.

The verifier constructs an exact, realizable eight-neuron example in three
orthogonal role subspaces.  Q has per-role AR(1) correlation with `r=7/8`;
each matched control has a star correlation geometry.  Both have zero role
means and identity per-neuron 3x3 role Gram, so they satisfy the frozen matched
moments.  Their relaxed scores are identical, giving corrected relaxed score
and both uncertainty terms zero when duplicated across the frozen two experts
and eight controls.  The implementation therefore hard-kills.  Yet the legal
FP16 coefficients are only `7/8` and `49/64`, both exactly representable in
binary16, and independent replay gives:

- Q legal FP16 `s = 0.7995602818589078`;
- control legal FP16 `s = 0.5885652320580218`;
- legal corrected `s = 0.21099504980088601`;
- frozen required gross `s = 0.1858070514584381`.

There is an open neighbourhood of this construction with the same strict
inequality, so it is not a measure-zero rescue.  The gross Q-only relaxed gate
remains a sound hard kill; the control-corrected relaxed gate is only a
heuristic and cannot claim family elimination.

Repair: remove the control-corrected relaxed hard kill.  If a corrected hard
upper bound is desired, subtract a proved lower bound on the legal control
score—not an independently loosened control upper bound.  Always retain a
computed legal-FP16 survivor.

### FOSP2-FW-001 — unsealed import code executes before preflight

`_artifact_rows()` enumerates only top-level objects for which
`Path.is_file()` is true.  It ignores directories and directory symlinks.
The documented launcher uses ordinary `python -B`, which places the script
directory on the import path.  The verifier copies the exact ten producer
files to a temporary directory, adds only `json/__init__.py`, and proves both:

1. `_artifact_rows()` accepts the tree and all ten regular-file hashes remain
   exact; and
2. the injected package executes while the runner performs its top-level
   `import json`, before artifact, audit, authorization, source, or output
   preflight.

This also affects the calibration and authorization entrypoints.  Checking
closure from inside an already importing script cannot establish entrypoint
identity.

Repair: use an external minimal launcher to open and hash a directory-descriptor
snapshot, reject every non-manifest object (including directories, symlinks,
reparse points, sockets and FIFOs), execute the already-open runner bytes from
an immutable descriptor/memfd, use isolated safe-path mode with a closed
environment/import path, and import dependencies from an independently sealed
runtime closure.

### FOSP2-FW-002 — audit verifier is named but never opened

The authorization builder and production runner parse an audit manifest and
check the receipt digest plus the literal row name `verify_audit.py`.  They do
not open the verifier, check its digest against its bytes, enforce the audit
directory's exact closure, or execute it.  The verifier demonstrates that the
acceptance function succeeds for a two-row synthetic manifest when no verifier
file exists.  Canonical JSON hashes are integrity checks, not auditor
signatures.

Repair: bind an externally trusted exact audit-manifest digest, or open and
verify every manifest member from a held directory, enforce exact object
closure, and execute the verifier with a sealed interpreter.  Duplicate-key
rejecting JSON should be used throughout evidence and authorization parsing.

## What passed

- Exact producer closure, manifest, every file hash/byte count, source receipt
  canonical seal, protocol schema, fixed bindings, and five ASTs.
- Independent Fraction arithmetic for the all-pairs 3x3 regression identity
  and index direction.
- A formal target-wise containment argument plus exhaustive enumeration of
  every path through seven neurons for deterministic nonnegative score panels.
- Cycle-cover target/predecessor direction, weakest-incoming-edge cuts,
  segment orientation, and explicit bridge scoring.
- Direct FP16 residual replay and little-endian binary16 representability.
- Matched-control mean/Gram construction algebra and the implemented MC,
  delete-one-expert jackknife, and quadrature formulas.  `+3SE` with eight
  controls remains a heuristic uncertainty convention, not a distribution-free
  confidence theorem.
- Factoradic, header, coefficient, payload-rate, logical-read and cold-page
  arithmetic.  Maximum cold-page amplification is
  `1.0054349308378698x < 2x`; logical expert-frame read is `1.0x`.
- V1 counterexample replay: capture `1534`, residual `770/2304`, and
  `s=0.7906051829300244`; v2 correctly routes it to the direct stage.
- Producer tests and verifier locally: 17 run, 14 pass, 3 Windows-only skips;
  source verifier `PASS`, 219 checks.
- Disjoint RunPod/Linux source-only replay: all 17 tests pass, including held
  output-parent replacement, symlinked evidence component rejection, and valid
  held-directory evidence; source verifier `PASS`, 219 checks; hashes equal.

## Zero-access boundary

No binding relative path was resolved or followed.  No model, Qwen, pinned,
validation, production-result, or external payload file was opened.  The audit
does not import CuPy/NumPy/SciPy/Torch, call CUDA/GPU APIs, issue a runtime or
production authorization, or run a payload.  SSH/SCP touched only the exact
enumerated source files in the disjoint RunPod scratch
`/var/tmp/int2_fosp_v2_source_audit_a7dc083b/producer`.

## Verification

From this directory, with bytecode disabled:

```text
python -B -I verify_audit.py
```

Expected status is `BLOCK_CONFIRMED`.  `AUDIT_SHA256SUMS.txt` seals the four
audit artifacts other than itself; both JSON receipts carry canonical unsigned
SHA-256 seals.
