# Independent hostile audit: RENORM-Q tiny-cell v0

Target manifest:
`340ba1f1c435be9cbfc58c75607cc5c5e07e6bb10692265b5938ebf530d926b9`

Target source root:
`8c1682ea514e067c4ba10b1e010abf1766cb842ea525233f7eb854896da6cac4`

## Verdict

**BLOCKED FROM PAYLOAD CAPABILITY PENDING SOURCE FIXES.**

The authenticated source closure and its eight tests pass.  Independent
randomized tests also confirm that min-sum and exhaustive enumeration agree
for valid normalized probability tables, including a depth-two tree.

Four blockers prevent treating the reported `modeled_bits` or decision status
as a safe compression gate:

1. Root, transition, and leaf NLL arrays are checked only for nonnegativity.
   They are not checked for Kraft normalization.  The target accepts a model in
   which all 16 four-symbol sequences cost zero bits, with Kraft sum 16.
2. `collective_variable_census` accepts an arbitrary caller-supplied map whose
   self-declared descriptor is zero.  A source-derived truth table obtains one
   bit of MI for free, bypassing the claimed frozen bank.
3. Several binary-alphabet members of the public map bank declare unreachable
   states.  They pass `MapSpec.validate` but fail the uniform-fiber leaf-model
   builder.
4. The frozen prose says a control-corrected lower-confidence gain below
   `0.03 bpw` hard-kills.  `kill_decision` compares the LCB only with zero and
   can promote a result whose LCB is `0.01 bpw`.

The RSMI per-weight arithmetic is otherwise correct for disjoint cells: MI and
state entropy are divided by sites per cell, and the default map descriptor is
amortized over all sampled sites.  This does not charge overlapping tilings,
traversal/beta search, probability tables, or repeated model selection.

The logical common/private read formula is correct, and all authority flags are
fail-closed.  There is no physical packet, model-table ledger, page ledger, or
expert-local decoder yet; the target discloses those limitations, so this is
not itself a source-only blocker.

No Qwen/model payload, GPU, network, or deployment operation was used.  The
target package was not modified.

Run:

```powershell
C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  research\renorm_q_smallblock_v0_20260904_independent_hostile_audit\hostile_audit.py
```
