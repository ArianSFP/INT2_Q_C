# Grouped-v5 layout-overlay v5 independent source audit

Verdict: **BLOCK**. This package authorizes neither calibration nor production.

The v5 successor substantially repairs the v4 findings at each instant that it
checks them. It binds raw entrypoint spelling, full component/object/mount
identity, rejects literal direct script execution, checks lexical/object/mount
disjointness, and completes workspace/auxiliary closure before creating the
production output root or journal. The frozen scientific plan and physical
ledger also replay exactly.

Three release blockers remain:

1. The source-trace module exposes an argv-only `main` that an external process
   can import and call without the authenticated bootstrap. The literal
   `__main__` guard does not execute on import, and no bootstrap capability is
   required by the operation.
2. Boundary revalidation and create-new are separate. The create helper reopens
   an absolute pathname after the check and protects only its final component.
   A swapped ancestor or mount can redirect the mutation between those calls.
3. The bootstrap closes the descriptors whose bytes it authenticated and then
   performs normal pathname imports. It therefore does not execute the exact
   authenticated byte buffers; a transient substitution can escape the before
   and after hash checks.

The independent verifier is pure PowerShell. It authenticates the exact
17-member producer closure, the lock's placeholder-normalized seal, sixteen
named source dependencies, source control flow, arithmetic, physical ledger,
manifest mutations, synthetic alias cases, and the three adversarial traces.
It performs no target-data, network, numeric-runtime, accelerator, calibration,
or production operation and does not modify the frozen v5 package.

Run with PowerShell 7 or later:

```powershell
./run_source_audit.ps1 -Producer /absolute/path/to/initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5
./verify_audit.ps1 -Producer /absolute/path/to/initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5 -Replay
```

Preserve v5 byte-for-byte. Implement repairs only in a distinct successor and
obtain a fresh independent source audit before any calibration or production
authorization.
