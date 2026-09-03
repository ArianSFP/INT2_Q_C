# Local RTX 3060 kill-first lane — 2026-09-03

## Inventory and executed result

- GPU: NVIDIA GeForce RTX 3060, compute capability 8.6.
- VRAM: 12,288 MiB total; 10,955 MiB free at inspection.
- Driver: 560.94.
- `nvidia-smi`: available at `C:\Windows\System32\nvidia-smi.exe`.
- A workspace runtime was subsequently provisioned at
  `C:\INT2__compression\.venv-cupy\Scripts\python.exe`: CPython 3.12,
  NumPy 2.5.2 and `cupy-cuda12x` 14.2.0 with toolkit components.
- CUDA toolkit / `nvcc`: absent. The standard CUDA toolkit directory is also
  absent.
- CuPy's diagnostic still reports `MSVCP140.dll -> not found` when launched
  without Windows DLL registration. The reproducible bounded lane now uses a
  task-local compatibility copy at
  `C:\INT2__compression\.tools\cuda_dlls_3060\MSVCP140.dll` (575,056 bytes,
  SHA-256
  `a4c2229bdc2a2a630acdc095b4d86008e5c3e3bc7773174354f3da4f5beb9cde`)
  and explicitly registers that directory plus the wheel-provided
  `nvidia\*\bin` directories with `os.add_dll_directory`. The file was copied
  from NumPy's bundled, content-addressed MSVC runtime; it was not installed
  machine-wide.
- The wheel-supplied CUDA DLLs exist under the environment's
  `site-packages\nvidia\*\bin` directories, but CuPy does not discover a
  `CUDA_PATH`.

The first mandatory GF(2) attempt failed before launch because the CuPy core
extension could not load its DLL dependencies.  That failure is retained as
diagnostic history, not as the final runtime result.

A bounded process-local diagnostic then registered the wheel CUDA DLL
directories and an existing system WinSxS Microsoft-runtime directory using
`os.add_dll_directory` (no files copied or installed). This allowed:

```text
CuPy import: pass (14.2.0)
device enumeration: pass (NVIDIA GeForce RTX 3060)
CUDA runtime: 12090
device allocation: pass (4,096 bytes)
first trivial elementwise kernel: did not complete
```

The apparent NVRTC/JIT hang was caused by the inherited/default kernel cache,
not a persistent compiler failure.  With a new task-specific `CUPY_CACHE_DIR`
the one-element RawKernel returned the exact value `42` in `1.4803 s`.

The corrected process-local launch then produced two successful source-free
results:

1. The exact GF(2) recurrence suite passed all `11/11` tests in `1.393 s`,
   including the mandatory CuPy/CPU Berlekamp--Massey parity test and hostile
   integral-scalar ABI cases.
2. The RM(5,12)^6 exact-cost CuPy smoke completed two greedy steps in
   `6.3226312 s`.  Both measured distortion deltas matched the CPU calculation,
   the trajectory was monotone, and literal packet replay passed.  Its packet
   was only `768 bytes = 1.5 bpw`, so the result is correctly classified as
   `PASS_LOCAL_GREEDY_MECHANISM_HOLD_PRODUCTION_SEARCH` and is not target-rate
   or Qwen evidence.  The receipt is stored outside the sealed source package
   at `C:\INT2__compression\.tools\local_rm6_steps2_receipt_20260903.json`.

No local Qwen/model payload was opened in either successful test.

Three further bounded CuPy checks now pass on the same process-local runtime:

3. The repaired Ramanujan-384 atomic v3 source-free fixture completed in
   `19.5759 s`.  Its CPU and RTX-3060 decodes produced the identical composite
   SHA-256
   `8d8e3e87ab8bd28d8b6f9195eabd61c74965fd0bc34e7c5c43b6adc4f62c5686`
   at exactly 245,760 bytes / 786,432 synthetic weights / 2.5 bpw.  This is a
   mechanism and CPU/CuPy parity result, not Qwen evidence.
4. The frozen same-layer common-latent aperture passed its complete local
   source suite: 15 tests passed and the Windows privilege-dependent symlink
   test was skipped.  Both CuPy count/objective checks matched CPU exactly,
   and the dedicated fixture receipt reported
   `PASS_SOURCE_FREE_CPU_CUPY_PARITY`.  The payload entrypoint remained at its
   compile-time HOLD.
5. The source-frozen CBIB-1 clustered same-layer aperture passed independent
   CPU/RTX-3060 parity. Counts and assignments were byte-identical and the
   maximum floating-point objective difference was
   `9.094947017729282e-13` bits. This was source-free mechanism evidence only;
   no Qwen payload was opened during that replay.

The label-flexible LOGIC-Q rank-0/rank-1 GF(2) source-free gate was also run
on the RTX 3060.  It completed in about 76 seconds, emitted a canonical
2.1500244140625-bpw synthetic packet with `1.007740495x` cold-read
amplification, and correctly hard-killed its deliberately difficult source at
`F=39.60912417` before controls.  This confirms execution and early stopping;
it is neither Qwen evidence nor a reason to close higher-rank, Reed--Muller,
ROBDD, or QTT label-flexible families.  The external receipt has SHA-256
`60eb07d2c4dba2629e023af6f3a7c65e9da9b76eabcd43be1718b84eddf9f7fb`.

## Existing bounded fixtures

The most useful first local tests, once a runtime exists, are:

1. `research/strata_gf2_recurrence_qwen_aperture_v0/test_source_only.py` —
   exact batched GF(2) BM kernel versus CPU, with no payload access. Use small
   fixed batches on Windows to stay comfortably below the display-driver TDR.
2. `research/strata_rm6_label_flexible_gate_v0/cupy_soft_search_smoke.py` —
   a 4,096-value synthetic, exact-cost RM6 local-search smoke. Start with one
   or two steps.
3. `research/logic_q_gf2_kill_gate_v1/run_source_free.py` — broader synthetic
   GF(2) source/control gate; defer until the two smaller kernel checks pass.

A cached audited STRATA artifact exists at
`results/qwen/strata_expert_affine_checkpoint/strata_expert_affine_n20n21.bin`
(8,847,360 bytes), with its independent audit receipt. It was only enumerated
and size-checked in this inventory, not opened. It is not needed for the first
two source-free tests.

## Bounded runtime procedure now working

The attempted Microsoft redistributable installation did not complete. The
bounded mechanism lane instead uses the task-local authenticated runtime copy
described above through process-local DLL registration. Every invocation must
also set `CUDA_PATH` to the wheel's `nvidia\cuda_runtime` directory and use a
fresh task-specific CuPy cache. This is a reproducible research workaround,
not a claim that the machine-wide CuPy installation is repaired.

## Authenticated local same-layer panel

The 32 layer-15 Up/Down panel members used by the prior same-layer experiment
were mirrored from the authorized RunPod to
`C:\INT2__compression\qwen_weight_cache\rd_structure_diag_cross_expert` for a
future independently authorized RTX-3060 payload replay. The mirror contains
exactly 100,663,296 bytes. Every member's length and SHA-256 matches the frozen
`same_layer_common_latent_entropy_gate_v0_deployment_20260903/panel_lock.json`.
The copy operation and hash verification did not run a codec or inspect weight
values. A separate local payload capability is still required before CBIB-1
may open these files.

The environment has now passed, in order, CuPy import, device-memory
allocation, a synchronized one-element RawKernel, the exact GF(2) kernel
parity suite, and the two-step RM6 search.

After that environment is supplied, the bounded commands are:

```powershell
$env:REQUIRE_CUPY_BM_TEST = "1"
& C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  C:\INT2__compression\INT2_Q_C\research\strata_gf2_recurrence_qwen_aperture_v0\test_source_only.py

& C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  C:\INT2__compression\INT2_Q_C\research\strata_rm6_label_flexible_gate_v0\cupy_soft_search_smoke.py `
  --steps 2
```

Both are source-free and should be stopped if either exceeds five minutes.
For any future full GF(2) payload aperture on this display GPU, start at
`--gpu-batch 128`; increase only after measuring kernel duration below the
Windows TDR margin. Do not begin a Qwen payload run merely because these
mechanism tests pass.
