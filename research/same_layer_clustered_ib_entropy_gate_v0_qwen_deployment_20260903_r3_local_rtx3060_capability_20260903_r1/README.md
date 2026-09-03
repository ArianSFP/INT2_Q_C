# CBIB-1 r3 local RTX 3060 one-use capability

Status: `AUTHORIZED_NOT_EXECUTED`.

This source-only package adapts only the frozen r3 deployment boundary to the
authenticated local Windows mirror.  The original r3 `run_gate.py`, CBIB core,
CuPy worker, panel, controls, thresholds, and Qwen file hashes remain in the
unchanged deployment package with manifest SHA-256
`5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f`.

The bridge uses the documented process-local runtime: it registers the pinned
compatibility directory and eight wheel `nvidia\*\bin` directories with
`os.add_dll_directory`, sets the exact wheel CUDA runtime as `CUDA_PATH`, and
uses a fresh fixed task-specific `CUPY_CACHE_DIR`.  It authenticates all ten
wheel RECORD closures before importing CuPy, then requires the pinned RTX 3060
name, UUID, compute capability, CUDA runtime/driver APIs, and driver string.
Nothing is installed or changed machine-wide.

The outer claim is created atomically before any capability, deployment,
runtime, GPU, output, or payload validation.  The original r3 child creates a
second atomic claim before it imports the worker or opens any Qwen payload.
Stdout, stderr, both claims, the result, and the final wrapper status are never
removed.  A failed or interrupted attempt is consumed and may not be retried.

Source-only tests intentionally exercised only HOLD paths and a temporary
claim-before-validation failure.  They did not import NumPy/CuPy, initialize a
GPU, enumerate or open the Qwen mirror, run the codec, use the network, stage a
run, or create any production claim/cache/result path.

For the sole eventual run, replace `<PUBLISHED_SOURCE_MANIFEST_SHA256>` in the
argv recorded by `AUTHORIZED_LOCAL_QWEN_ONCE.json` with the externally
published package-manifest digest and invoke that argv using
`subprocess.run(argv, cwd=recorded_cwd, shell=False)`.  Do not invoke it during
source review.  Any result remains an ideal label-entropy census, not a finite
codec or MSE result, until separately audited.
