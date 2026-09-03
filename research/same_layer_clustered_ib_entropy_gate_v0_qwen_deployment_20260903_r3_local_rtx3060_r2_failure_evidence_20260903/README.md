# CBIB-1 r3 local RTX 3060 r2 consumed-attempt evidence

Status: `FAIL_RUNTIME_GPU_IDENTITY_NO_QWEN_ACCESS`.

The sole r2 invocation is consumed. The outer claim, wrapper status, empty
stdout, and complete stderr traceback are retained at their original paths.
The traceback terminates in `canonical_uuid` because CuPy 14.2 returned a
19-byte Windows UUID buffer while r2 required exactly 16 bytes. The empty CuPy
cache and presence of the traceback prove runtime/GPU initialization began.

The frozen `run_gate.py` calls `_validate_runtime()` before it constructs the
payload/output paths or creates the inner claim. The runtime exception therefore
occurred before any Qwen file was enumerated or opened. The run root contains
only `child_stdout.jsonl` and `child_stderr.txt`; `ONE_USE_CLAIM.json` and
`result.json` are absent. There is no scientific result and no retry authority.

