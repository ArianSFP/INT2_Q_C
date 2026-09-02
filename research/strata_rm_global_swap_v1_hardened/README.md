# Hardened global STRATA RM-order swap v1

Date: 2026-09-02

Status: **frozen source-only successor; execution and all payload authority held**.

This sibling repairs the authority findings in the independent audit of
`strata_rm_global_swap_v0` without changing that frozen package.  It pins:

- v0 source root
  `4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c`;
- independent-v0 audit root
  `7eabe4580908d4a79eceb2f7fdaf838d535028c06263c2f4841032664db11ad0`;
- the current base encoder, BEC wrapper, and independent STRATA decoder by
  their exact SHA-256 values.

No Qwen, Gaussian-control, container, coarse, or model-weight payload was read
while producing this package.

## What is hardened

### Exact source closure

`authority.authenticate_flat_package` and `verify_source.py` enumerate every
top-level directory entry.  Only the exact manifest plus exact regular files
are legal.  Symlinks, junction-like directory entries, nested `cupy/` or
`numpy/` packages, FIFOs, sockets, device nodes, and unmanifested files fail.

### Current hook integration

There is no `install(base, hook)` API.  `current_integration_worker.py` is
started as a fresh `python -I -B` process with an allowlisted environment.  It:

1. hashes the exact current base and BEC sources;
2. imports the base and wrapper itself from those paths;
3. requires `bg.base is base` and the live pre-swap hook to be exactly
   `bg.bec_flags`;
4. captures that exact reference and installs the RM-order replacement;
5. checks the final hook object after all imports;
6. permits only `N=2**20` and `N=2**21`;
7. has no payload CLI option and reports `payloads_opened=0`.

Thus a caller cannot inject a `SimpleNamespace`, alternative hook, copied
module, or later monkeypatch and call it an integrated result.

### Real CuPy

`real_cupy_worker.py` is also a fresh `python -I -B` child.  The parent strips
`PYTHONPATH`, `PYTHONHOME`, and user-site import state.  The child rejects
preloaded `cupy`/`numpy`, rejects origins inside the experiment, external, or
temporary roots, requires a live CUDA runtime/driver/device, executes and
synchronizes a GPU arithmetic probe, and compares the complete `2**20` and
`2**21` GPU orders to an independently constructed NumPy order.

This is trusted-runner process isolation, not cryptographic remote hardware
attestation.

### Literal physical-result authority

`physical_authority.validate_physical_bundle` has no parameters for packet
bytes, decoded values, rate, MSE, F, a decoder object, a hook, read
amplification, or a selected configuration.  It accepts only paths and an
out-of-band expected SHA-256 for a canonical experiment commitment.

For every committed case it:

1. opens a nonempty literal packet and exact source BF16 files under an
   authenticated external evidence root;
2. launches the hash-pinned independent decoder in a fresh `python -I -B`
   process under a fixed protocol;
3. reads literal FP64 reconstruction files;
4. requires the decoder's independently regenerated canonical packet to equal
   the original packet as bytes;
5. recomputes FP64 SSE, source energy, relative MSE, physical bpw, and
   `F = D * 2**(2R)` from those bytes;
6. validates an expert-local packet read trace and computes amplification;
7. in production mode, requires matched Gaussian cases through the identical
   pipeline, `2.15 <= R <= 2.5`, `F <= 0.8`, maximum reads `<2x`, and portable
   evidence over at least two named SwiGLU-MoE architecture families.

Production mode also authenticates an exact independent-audit source closure
for the decoder worker and binds that audit to the worker SHA-256.  A boolean
claim that a decoder is “independent” is insufficient.

The bundled `fixture_decoder_worker.py` exercises this authority protocol on a
small synthetic packet.  It cannot be used in production mode and confers no
Qwen, rate-distortion, read, matched-control, or universality evidence.

## Run boundaries

Static/source gate, after this package's manifest hash is externally pinned:

```bash
python -I -B run_source_gate.py \
  --package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v1_hardened \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --v0-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v0 \
  --v0-audit-package /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v0_independent_source_audit_20260902 \
  --external-root /workspace/INT2__compression \
  --output /tmp/strata_rm_global_swap_v1_source_gate.json
```

Add both `--run-current-integration --run-real-cupy` only on the trusted CuPy
RunPod.  The source-only hostile suite is:

```bash
python -I -B test_source_only.py
python -I -B hostile_tests.py
```

## Honest remaining holds

- The source and hostile tests have not yet executed in a Python environment.
- The isolated current-hook and real-CuPy workers have not yet executed.
- There is no physical `N=2**20`/`N=2**21` Qwen packet from this candidate.
- The pinned base encoder's public CLI still admits only its historical
  smaller-length range; a separately audited global payload launcher remains
  required even though the live current hook object is now authenticated.
- A production independent decoder worker implementing the fixed v1 protocol
  has not been frozen or independently audited.
- No exact BF16 Qwen score, matched Gaussian control, cross-family portability
  panel, or routed-read trace exists.

Consequently the only admissible current status is
`FROZEN_SOURCE_ONLY_UNEXECUTED__HOLD_RUNPOD_AND_PAYLOAD`.
