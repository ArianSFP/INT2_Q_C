# CBIB-1 r3 local RTX 3060 r1 consumed-attempt evidence

Status: `FAIL_ATTEMPT_CONSUMED_BEFORE_GPU_OR_QWEN_ACCESS`.

The single authorized r1 wrapper invocation created its immutable outer claim,
validated the sealed source/runtime prerequisites, created the empty run root,
and then raised `PermissionError` at the next operation: creation of the CuPy
cache beneath `C:\INT2__compression\.cupy_cache`.  The run root remains empty;
the cache and result are absent.  Because the frozen wrapper contains no delete
path, that retained state proves the child bridge was never launched.  It
therefore could not import CuPy, initialize the GPU, create the inner claim, or
open any Qwen payload file.

The r1 attempt is permanently consumed and must not be retried or cleaned up.
`FAILURE_EVIDENCE.json` binds the two original retained artifacts, the empty run
root, the absent cache/result, and the exact sealed r1 wrapper/capability source.
`verify_failure_evidence.py` is a read-only stdlib verifier for those claims.

