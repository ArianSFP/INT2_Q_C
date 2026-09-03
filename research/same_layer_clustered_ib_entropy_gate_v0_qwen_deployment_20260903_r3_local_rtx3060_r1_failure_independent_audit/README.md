# CBIB-1 r3 local RTX 3060 r1 consumed-failure audit

This is an independent, read-only audit of the single authorized local RTX
3060 invocation.  It grants no execution or successor authority.

The attempt is consumed.  The frozen wrapper wrote its `O_CREAT|O_EXCL`
outer claim, authenticated its prerequisites, created the fixed run directory,
and then recorded `PermissionError`.  The fixed run directory is empty, the
fixed CuPy cache path is absent, and all four child artifacts (stdout, stderr,
child claim, and result) are absent.  In the authenticated wrapper, cache
creation precedes child log creation and `subprocess.run`.  Consequently the
observed state places the failure at creation of the fresh CuPy cache, before
the GPU/Qwen child could be launched.

This proves no CBIB Qwen measurement was produced and no scientific conclusion
can be drawn from r1.  The parent-side prerequisite pass only authenticated
source/runtime files and resolved the fixed payload-root directory; its pinned
bridge explicitly does not import NumPy/CuPy or touch a GPU during that phase.

Run the verifier from any working directory with a standard-library Python:

```text
python verify_failure_audit.py
```

The verifier is read-only.  It accesses only this audit package, the frozen
capability package, and the four exact attempt paths named in the receipt.  It
does not import NumPy/CuPy, initialize a GPU, enumerate or open Qwen weight
files, use the network, or modify the consumed attempt.

