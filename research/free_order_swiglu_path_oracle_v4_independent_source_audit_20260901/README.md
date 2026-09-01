# Independent hostile source audit: FOSP-ARX v4

Verdict: **BLOCK for calibration, model access, and deployment**.  The frozen
v3 science and rate/read ledger pass, and the v4 package is honestly inert.
The proposed pre-Python boundary is not yet an authenticatable deployment
boundary.

This audit is independent of the unfinished/cancelled v3 audit directory.  It
compares the v4 scientific files directly with the immutable v3 producer
files and uses no v3 audit receipt or verdict.

## Blocking findings

### FOSP4-FW-001 — the runtime image authenticates itself

The externally pinned v4 contract fixes the launcher digest and bootstrap
digest, but it fixes no interpreter/runtime manifest digest or image ID.  A
launch request supplies both the runtime member hashes and the `image_id` that
is derived from those same untrusted rows.  The validator has no external
runtime pin argument.

The independent hostile fixture replaces `bin/python3.12` with hostile bytes,
updates that row's length and SHA-256, recomputes `image_id`, and leaves all
declarative immutability booleans true.  The exact sealed validator returns
`PASS_CONTRACT_VALIDATION_ONLY`.  This is internal consistency, not
authentication of an independently approved runtime.  The same issue applies
to startup modules such as `encodings` and to the assertion that four
synthetic files are a complete recursive runtime closure.

Repair: require a runtime-manifest digest or immutable-image identity supplied
by an external trust root and bind it into the pinned launch contract.  The
native launcher must open and hash the complete held object graph against that
external identity before Python startup.  The runtime manifest cannot mint
its own trust anchor.

### FOSP4-FW-002 — the native enforcement boundary is absent

The package deliberately contains no native launcher, runtime image, runtime
manifest, or deployment entrypoint.  `launch_contract.py` is explicitly an
inert Python model and trusts request fields such as `immutable`, ownership,
link count, complete closure, and authentication order.  Therefore the source
package provides no auditable evidence that a real launcher performs no-follow
opens, retains file identities across verification, establishes platform
immutability, closes descriptors/environment/namespaces, or starts Python only
after those operations.

The bootstrap digest is correctly frozen, but the request schema does not bind
the eventual bootstrap execution source to a held descriptor or held byte
snapshot.  It only carries the assertion
`authenticated_and_held_before_execution: true`.  Authentication of one held
copy followed by execution through a replaceable pathname would satisfy the
model while violating the intended boundary.

Repair: build and independently audit the externally pinned native launcher
and immutable runtime.  Its receipt must bind actual object/descriptor
identities and event order, and it must execute the already-authenticated
bootstrap bytes (for example from a sealed held descriptor), not reopen a
pathname.  This source-only audit cannot substitute for that deployment
audit.

### FOSP4-FW-003 — forbidden shell launchers pass validation

The contract says ordinary Python and shell launchers are forbidden.  The
validator rejects a basename containing `python` and the exact basename `sh`,
but accepts `/bin/bash`.  It also accepts a path containing a `..` component
as "canonical".  Both cases receive `PASS_CONTRACT_VALIDATION_ONLY` in the
independent fixture.

Repair: the external pin must name a separately audited native executable
identity, not infer implementation class from its filename.  Canonicalization
and no-follow object identity must be established by the native launcher and
trust root.  At minimum the source validator should reject every shell fixture
and non-normal path it claims to reject.

## What passed

- Exact producer closure: 11 regular, single-link files, 89,408 bytes;
  `PACKAGE_MANIFEST.json` SHA-256
  `9762c7edffc86d21f4400d4fac37ecab33c75ede6f7f9ab7c3ef5b95fe51e066`.
- `scientific_oracle_v3.py` and `scientific_protocol_v3.json` are byte-for-byte
  identical to v3 at SHA-256 `9ca6f4bd...a070` and `f4660cb8...097`.
- The sole family hard kill is the gross Qwen relaxed necessary bound.
  Corrected relaxed reuse is diagnostic-only, and legal FP16 is computed and
  required before the control-corrected decision.
- Independent n=8 replay: corrected relaxed `0`, Q legal FP16
  `0.7995602818589078`, control legal FP16 `0.5885652320580218`, and corrected
  legal `0.21099504980088601 > 0.1858070514584381`.
- Independent side ledger: 117,224 bits, `0.024843004014756944 bpw`;
  required gross saving `0.1858070514584381 bpw`; logical read `1.0x`;
  maximum frozen cold-page read `1.0054349308378698x < 2x`.
- The README correctly states that the native executable/runtime are absent,
  gives a regular absolute illustrative launcher path, forbids generic Python,
  shell, symlink and mutable-venv substitutions, and grants no authority.
- Producer replay on the already-present disjoint RunPod copy: verifier
  `PASS`, 105 checks; hostile suite 18/18.  Those tests catch inconsistent
  byte substitution, but not a hostile self-consistent runtime manifest or
  `/bin/bash`.

## Authorization boundary

No Qwen/model, pinned, validation, calibration, or production payload path was
resolved or opened.  This audit imports no NumPy, CuPy, Torch, CUDA, or model
library, performs no GPU/network operation, and issues no authorization.  The
science is eligible to be carried into a v5 repair, but v4 itself must not be
used to authorize source access or a GPU experiment.

## Verification

With a separately trusted standard-library Python used only for audit QA:

```text
python -B -I verify_audit.py
```

Expected status is `BLOCK_CONFIRMED`.  `AUDIT_MANIFEST.json` seals every other
member of this flat audit directory; authenticate its externally reported
SHA-256 before relying on the receipt.
