# PAIRPATH-P2 local RTX-3060 CuPy source-free preflight

## Outcome boundary

This package accelerates and tests only the optimistic PAIRPATH-P2
single-letter oracle mechanism. It cannot open a model payload: it has no
payload locator, production authority, network client, remote runner or model
library. `RUN_GATE.json` must remain at `HOLD_PRODUCTION_AND_PAYLOAD` or the
preflight fails before importing CuPy.

The backend mirrors the repaired PAIRPATH source semantics:

- four legal labels for each of two experts;
- Up and Down are decoder-visible and mutual information is conditioned on
  role;
- every role receives the same global multiplier
  `lambda * sum(UpDown^2) / (4*N)`;
- independent and joint models use the identical ordered bank of nearest,
  equal-label and 16 constant-pair starts;
- alternating optimization stops after at most eight updates;
- label ties select the lowest label, or the lowest flattened pair index;
- output includes raw points and convexified equal-rate/equal-MSE frontiers.

The preflight also loads the exact source-closed PAIRPATH-P2 r2 reference core
at SHA-256
`2c99a31aef669cabbb67137061233640b013e8c50a5132ddbcc9ffec2c239034`
and proves that all synthetic CPU winners, SSE values and rates are identical.
Its source manifest is pinned at
`21983efff5ac5c0593a655cae4136d35ca24400fd807f9fe4be458a34b18e622`.

The `[2,N,4]` FP64 distortion table remains resident on the GPU. Label updates
materialize at most `[chunk,4]` independent or `[chunk,16]` joint costs, and
counts are exact `int64` CuPy `bincount` results. Final candidate scores use the
canonical NumPy calculation. This makes the accelerated update the dominant
work while preventing a different GPU reduction tree from changing a close
multistart winner.

## Mandatory tests

The suite covers:

1. deterministic symmetric-start ordering and deduplication;
2. role-conditioned MI (one 2-bit aligned role and one zero-MI role);
3. a single global Up/Down bit weight despite strongly unequal role energy;
4. exact lowest-index tie behavior, including a non-multiple chunk tail;
5. independent and 16-way joint label/count parity over three multipliers;
6. complete CPU/CuPy RD-point and convexified-gate parity.

The preflight additionally benchmarks complete CuPy update-plus-count calls
and reports conservative explicit-array memory plus a worst-case linear timing
estimate for the staged 1/64 and 1/8 apertures of a 768-by-2048 role.

## Invocation

From the repository root:

```powershell
& C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  research\pairpath_p2_local3060_cupy_preflight_v0\run_source_free_preflight.py
```

The script registers the authenticated task-local MSVC runtime and the
wheel-provided NVIDIA DLL directories with `os.add_dll_directory`, sets the
wheel CUDA runtime as process-local `CUDA_PATH`, uses a task-specific CuPy
cache, and requires exactly GPU UUID
`GPU-458a424a-76e3-65e5-0470-803e0ed131ca`.

## Non-claims

Passing proves source-free CPU/CuPy mechanism parity on the pinned local GPU.
It is not weight evidence, a Qwen result, a physical packet, a production seal,
or permission to run a payload. A separately reviewed and authorized package
would still be required for any model data.
