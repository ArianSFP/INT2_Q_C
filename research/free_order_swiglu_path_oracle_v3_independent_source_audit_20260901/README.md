# Independent hostile source audit: FOSP-ARX-v3-DIRECT-SEALED

Verdict: **BLOCK**. The v3 scientific repair is correct: gross target-wise
reuse contains every legal exact path; the control-corrected relaxed statistic
is diagnostic-only; and the legal-FP16 statistic is computed before the only
control-corrected decision. The exact n=8 counterexample remains a legal FP16
survivor. Deployment is nevertheless blocked by two pre-execution trust gaps.

## Blocking findings

### FOSP3-FW-001 — the package bootstrap cannot authenticate itself

The documented launcher executes `bootstrap_v3.py` by pathname. That file is
inside the package manifest which it checks only after Python has already
executed its bytes. In a copy of the exact package, the verifier replaces only
`bootstrap_v3.py`, supplies the original externally pinned manifest digest,
and observes the replacement write a sentinel and counterfeit the documented
snapshot PASS. The original manifest and every other package file remain
unchanged. `verify_package.py` has the same circular self-binding defect when
launched directly.

An external digest argument is not a trust anchor for code that has already
begun executing. V4 needs an externally pinned native launcher, or equivalent
external trust anchor, which opens and hashes the bootstrap before execution
and launches held/sealed bytes rather than the unchecked pathname.

### FOSP3-FW-002 — filesystem runtime code executes before runtime closure

On the audited Windows Python 3.12.13, `-I -S` still reaches the script with
`encodings/__init__.py`, `encodings/aliases.py`, `encodings/cp1252.py`, and
`encodings/utf_8.py` already loaded from filesystem paths. The bootstrap does
not purge inherited `sys.modules`. A sealed probe whose declared synthetic
runtime contains no `encodings` file successfully imports that inherited
filesystem module after the normal filesystem finder is removed.

Hashing `sys.executable` from Python and retrospectively checking a runtime
tree therefore cannot establish that no unverified startup code ran. The
interpreter's startup-time encodings/runtime tree and native loader
dependencies must be independently immutable and authenticated before Python
starts.

### FOSP3-DOC-001 — generic `python` is not the required interpreter identity

The README examples use generic `python`. A symlinked interpreter path is
correctly rejected by the bootstrap's regular-file check (the parent audit
reported this on Linux, but that unsealed message is deliberately excluded
from formal evidence). Release instructions must name the canonical regular
interpreter whose bytes were externally pinned. This documentation defect is
subordinate to the two firewall blockers above.

## Scientific findings that passed

- Independent exact-Fraction all-role 3x3 regression, target/predecessor index
  direction, trace/capture identity, direct residual identity, and a separate
  little-endian binary16 replay.
- Formal target-wise proof plus exhaustive enumeration through every path for
  deterministic nonnegative score matrices of sizes 2 through 7.
- Cycle-cover predecessor direction, weakest-incoming-edge cuts,
  deterministic segment order, and explicit bridge scoring.
- Exact n=8 construction: three orthogonal roles in the zero-coordinate-mean
  subspace; Q AR(1) with `r=7/8`; matched-control star geometry with
  `rho=r^2=49/64`; both coefficients exactly binary16. Recomputed scores are
  Q legal `0.7995602818589078`, control legal `0.5885652320580218`, corrected
  legal `0.21099504980088601`, versus required gross
  `0.1858070514584381`.
- AST and dynamic call-trace proof that the only hard-kill literal is the
  gross-Q relaxed necessary bound. Corrected legal FP16 is evaluated first;
  changing the corrected relaxed diagnostic cannot change the decision.
- Factoradic, header, coefficient, payload-rate, logical-read, and cold-page
  arithmetic. Total side information is 117,224 bits; logical read is `1.0x`;
  maximum cold-page amplification is `1.0054349308378698x < 2x`.
- Exact package directory and ordinary extra-file rejection. The Windows host
  could not create symlink/reparse fixtures and has no filesystem FIFO/Unix
  socket support; these cases remain sealed for Linux replay and do not affect
  the already-proved BLOCK.

The exact direct runner exits deployment-blocked under `-I -S`. A hostile
plain direct launch with a package-local `math.py` executes that module before
the source-only guard, confirming that the oracle is not itself a safe
entrypoint.

## Evidence and zero-access boundary

The verifier opens from held file descriptors, identity-checks, hashes, and
enforces exact closure for all ten v3 package objects, all ten immutable v2
package objects, all five sealed v2 audit objects, and all four objects in this
audit. It parses the v2 BLOCK receipt and opens the v2 source containing the
superseded hard kill. Evidence is never accepted merely by row name.

No source binding was resolved or followed. No model, Qwen, pinned panel,
payload, result, runtime/calibration evidence, or authorization manifest was
opened. No NumPy/CuPy/SciPy/Torch/model module was imported; no CUDA/GPU,
network, remote, calibration, or authorization action occurred. The parent
agent's unsealed Linux status message is not formal evidence in this audit.

## Verification

First use an independently trusted hash tool to pin `AUDIT_SHA256SUMS.txt` and
the `verify_audit.py` row before executing the verifier. Then run:

```text
<canonical-regular-python> -B -I verify_audit.py \
  --audit-manifest-sha256 <externally-pinned-audit-manifest-sha256>
```

Expected status is `BLOCK_CONFIRMED` with the receipt's exact check count.

Safe isolated Linux replay, after copying only the four sealed audit objects,
the exact v3 package, exact v2 package, and exact sealed v2 audit into a fresh
nonshared scratch tree and externally checking all three source manifests:

```text
env -i PATH=/usr/bin:/bin /usr/bin/readlink -f /usr/bin/python3
env -i PATH=/usr/bin:/bin <canonical-regular-python> -B -I verify_audit.py \
  --audit-manifest-sha256 <externally-pinned-audit-manifest-sha256>
env -i PATH=/usr/bin:/bin <canonical-regular-python> -B -I \
  ../free_order_swiglu_path_oracle_v3/test_source_only.py
env -i PATH=/usr/bin:/bin <canonical-regular-python> -I -S \
  ../free_order_swiglu_path_oracle_v3/bootstrap_v3.py \
  --package-manifest-sha256 8584bde5c09fb7df531884c2100d3892dd1e12fbe689cf5d23ce091918d96470 \
  --verify-package
```

Do not substitute `/usr/bin/python3` until `readlink -f` proves and external
hashing pins the canonical regular target. Do not add source/model/runtime
evidence, GPU visibility, network access, or authorization inputs.
