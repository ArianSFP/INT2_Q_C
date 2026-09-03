# CBIB-1 r3 local RTX 3060 r2 consumed-failure audit

This package is an independent, read-only audit of the single authorized local
RTX 3060 r2 invocation at the fresh `R2_09F4C6D1` paths. It grants no execution,
retry, repair, deletion, or successor authority.

## Sealed conclusion

The r2 attempt is consumed and cannot be retried. The outer wrapper created its
durable `O_CREAT|O_EXCL` claim, authenticated its frozen packages, created its
fresh run and CuPy-cache directories, and launched the shell-free child.

The child imported CuPy and initialized the CUDA runtime far enough to query
the runtime version, driver version, current device, device properties, and raw
device UUID. It then failed at the bridge's exact `CUDA UUID bytes` validation.
This is a runtime-boundary failure, not a GPU-free preflight failure.

The frozen `run_gate.py` calls `_validate_runtime()` before it constructs the
payload-root `Path`, creates its inner `ONE_USE_CLAIM.json`, loads the worker, or
calls `run_authorized_panel`. Because the traceback terminates inside that
runtime validator:

* no Qwen weight file was enumerated or opened by this attempt;
* the inner `ONE_USE_CLAIM.json` is absent;
* `result.json` is absent;
* child stdout is empty; and
* no scientific result exists.

The outer claim and authority status remain present. The sole r2 authority has
therefore been consumed despite producing no payload result.

## Verification

Run with a standard-library Python from any working directory:

```text
python verify_failure_audit.py
```

The verifier is read-only. It hashes the sealed source/evidence, parses Python
source, and checks only the exact r2 attempt paths. It does not import NumPy or
CuPy, initialize or query a GPU, enumerate Qwen weights, access the network,
modify the attempt, or grant a successor authority.

