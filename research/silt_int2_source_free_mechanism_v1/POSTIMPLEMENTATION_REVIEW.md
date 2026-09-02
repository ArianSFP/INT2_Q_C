# SILT v1 postimplementation review

Status: **implementation complete; stop for independent audit**.

This review is intentionally not a manifest, source seal, result receipt,
payload authorization, or universal-codec promotion. No model or Qwen payload
was discovered, opened, enumerated, statted, hashed, copied, or decoded. No
canonical synthetic result was generated.

## Review outcome

The v0 audit required a format-version break. V1 now has:

1. exact owner-aware byte conservation, strict rational cold decisions, and an
   independent instrumented page union;
2. a page-aligned external directory supporting 1 through 256 experts and
   different geometry for every expert;
3. caps checked before factorial, product, directory loop, prefix sum, slice,
   conversion, or decoded allocation;
4. a 30-bit physical arithmetic guard, hard meaningful-bit EOF, exact reader
   exhaustion, minimum payload bytes, and mandatory ordinary-decode re-encode;
5. six canonical GF(2) transforms with IDs 6/7 rejected, while Z4 retains all
   eight maps;
6. a stdlib-only, content-addressed, auth-before-import private snapshot;
7. exclusive no-follow staging, file and directory fsync, atomic
   `renameat2(RENAME_NOREPLACE)`, and descriptor-based publication rehashing;
8. mandatory CuPy search with exact logical transfer bytes, CPU/GPU equality,
   host RSS, baseline-subtracted NVML process/device VRAM, CuPy pool, GPU UUID,
   and asserted CUDA-logical-to-NVML-physical PCI mapping; and
9. a separately implemented parser/decoder/re-encoder with the same bounds,
   canonical arithmetic language, GF(2) policy, and owner ledger.

## Source-only execution evidence

Before this review text was finalized, three content-authenticated clean-source
runs passed on the provided NVIDIA GeForce RTX 5090. The latest pre-review run
executed 15 tests in 11.753 seconds with zero failures, errors, or skips. The
suite included runtime containers with 1, 128, 249, 250, and 256 experts and
different lane/vector shapes, plus independent full decode and byte re-encode.

The latest canonical `2048/1024 × 97`, eight-candidate GPU search measured:

- exact logical H2D: 305,832 bytes;
- exact logical D2H: 2,483,200 bytes;
- model H2D: zero bytes;
- CUDA-event H2D: 3.209 ms;
- CUDA-event kernels: 100.167 ms;
- CUDA-event D2H: 1.553 ms;
- CPU reference scoring: 7,303.922 ms;
- search wall time: 7,690.497 ms;
- host RSS baseline/peak/delta: 334,876,672 / 392,179,712 / 57,303,040 bytes;
- NVML current-process VRAM baseline/peak/delta:
  522,190,848 / 528,482,304 / 6,291,456 bytes;
- NVML total-device VRAM baseline/peak/delta:
  1,053,949,952 / 1,060,241,408 / 6,291,456 bytes; and
- CuPy-pool peak delta: 1,780,736 bytes.

The mapped device was CUDA logical 0 at PCI `0000:16:00.0`, NVML physical 0,
UUID `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722`, compute capability 12.0,
driver 580.126.09, CUDA runtime 13.2, CuPy 14.2.0, NVML library
13.580.126.09, and `nvidia-ml-py` 13.610.43. Every value above came from the
runtime receipt; transfer sizes are exact logical array bytes, not claimed
physical PCIe transactions.

The final authenticated rerun necessarily occurs after editing this review,
because this file is itself a root member. Its observed root and stdout receipt
must remain external observations rather than an embedded self-authentication.

## Hostile gates covered

- The audit layout `E=8, G=8192, F=[4096,8192,...]` produces exactly `12/5`
  for expert zero and fails the strict gate.
- An actual v1 container produces unequal `[4096,8192,...]` expert frames;
  every cold numerator equals an instrumented page union.
- Changing bytes owned by other experts cannot change one expert’s owner share
  or amplification. Exactly `2×` fails.
- Expert counts 0, 257, and `2^32-1`, and forged lane/vector values
  `2^32-1`, reject promptly with controlled format errors.
- Short binary strings are exhaustively checked under balanced and
  `[1,65535]` Q16 rows. Every successful decode exhausts exactly the declared
  bits. Truncation at every meaningful boundary, ±1 lengths, 0/1/31 bits,
  appended/deleted payload bytes, nonzero guard, byte tails, and valid-hash
  GF(2) aliases reject in ordinary decode.
- Extra, missing, symlinked, or modified source members and poisoned cwd,
  `PYTHONPATH`, and `sitecustomize` paths reject or remain unexecuted before
  imports.
- Publication tests cover existing targets, output and parent symlinks,
  failures after artifact/index/completion fsync stages, and two simultaneous
  publishers; no partial final tree appears and exactly one racer succeeds.

## Remaining scientific and authority limits

- This is still a constructed finite-label mechanism. It supplies no evidence
  of structure, rate, distortion, or MSE in any real model.
- It contains no float-to-label quantizer, Gate/Up/Down adapter, raw-weight MSE
  scorer, 2.15–2.5 bpw universal packet, or held-out model-family result.
- The finite arithmetic coder is a transparent reference backend, not a
  production rANS throughput claim.
- NVML and host RSS peaks are 2 ms samples. Exact transfer-byte accounting is
  independent of the sampler, but a shorter transient could fall between peak
  samples.
- CPU finite scoring dominates this deliberately auditable search prototype.
- The producer does not publish or freeze an external source root. That action
  belongs to a separate auditor, as does any canonical result execution.

## Boundary decision

Producer verdict: **READY FOR INDEPENDENT SOURCE-FREE V1 AUDIT, NOT READY FOR
PAYLOAD ACCESS OR CODEC PROMOTION**. Stop here. An external auditor must compute
and publish the source root, replay the isolated verifier, inspect the complete
stdout/stderr/exit receipt, and independently challenge the format and ledgers.
