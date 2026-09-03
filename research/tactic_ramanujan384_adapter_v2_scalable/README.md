# TACTIC Ramanujan-384 scalable authority v2

Date: 2026-09-02

Status: **CPU source-only checks passed; CuPy target fixture pending; no Qwen
or coarse-model payload authority**.

V2 repairs the operational holds in the independent v1 review while retaining
literal coarse/fine replay and original-source FP64 scoring. It is a source
architecture, not a Qwen result.

## Batched encoder

`scalable_core.py` is static: it has no dynamic module loading and no scalar
device synchronization. For each role it computes block-specific correlations
and valid-coordinate atom norms, selects fourteen atoms, materializes every
rank-prefix linear system in one tensor, and calls one batched solve. All
canonical binary16-scale/int11 coefficient candidates are reconstructed with
one `B x 14 x 4096` einsum and scored directly on valid source coordinates.
Only winning metadata and the final summary cross to the host, in bulk.

The packet parser proves that emitted winner records reproduce the selected
coefficient state. The dictionary is built once and reused by three source
roles, literal replay, phase controls and all eight Gaussian controls.

## Canonical Gaussian control

V1's twelve-uniform Irwin--Hall control is gone. V2 hashes absolute integer
counters with SplitMix64, converts the top 53 bits to exact midpoint uniforms,
and applies the mathematical Box--Muller transform. Results are rounded once
to IEEE binary32 and widened exactly to binary64. Those serialized bytes are
generated on the host once and copied unchanged to NumPy or CuPy, so backend
RNG and backend transcendental implementations cannot change the control.

This is a deterministic finite Monte-Carlo sample from the Gaussian
construction; no finite PRNG is claimed to have continuous support.

## Target-rate source-free fixture

The sealed fixture uses `[128, 2048]` roles:

```text
weights        786432
coarse bytes   235776
fine bytes       9216
header bytes      512
physical bytes 245760
physical rate      2.5 bpw
```

Exact public Ramanujan atoms make the synthetic source pass the absolute gate,
so the CuPy runner must execute one phase and eight complete Gaussian-control
searches. It also byte-compares the canonical Gaussian host array after a
CuPy round trip.

## Coarse decoder and closure

The coarse decoder is no longer an unbound callable. Before use, v2 opens and
hashes its runtime class source, decoder source manifest, independent auditor
manifest, capability record and independent PASS receipt. The literal decoder
receives only coarse bytes and public geometry, and every role output must
match an independently recorded FP32 reconstruction hash.

The package caps block counts to inherited uint32 header fields, caps payload
lengths to uint64, and uses integer page rounding. Tail masks participate in
atom norms and Gram construction as well as SSE.

`../tactic_ramanujan384_adapter_v2_scalable_bootstrap_verify.py` is an external
bootstrap verifier. Its own SHA-256 is supplied explicitly; it loads no code
from this package before validation and rejects any extra, missing, symlinked,
or nested entry.

## Commands

```bash
python -I -B research/tactic_ramanujan384_adapter_v2_scalable/test_source_only.py

python -I -B research/tactic_ramanujan384_adapter_v2_scalable/run_source_free_cupy_target_fixture.py \
  --authorization RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_V2_TARGET_RATE_CUPY \
  --manifest-sha256 <manifest-sha256>

python -I -B research/tactic_ramanujan384_adapter_v2_scalable_bootstrap_verify.py \
  --package research/tactic_ramanujan384_adapter_v2_scalable \
  --manifest-sha256 <manifest-sha256> \
  --bootstrap-sha256 74f5a56f1371f67ffa4e83ea34b761c2de61ea0900e3374cc25092f2d333e92c
```

All commands are source-only. The RunPod runtime, independent hostile v2
review, real coarse-decoder capability, Qwen payload, storage telemetry and
HBM measurement remain explicit holds.

The local Python 3.12.13 / NumPy 2.3.5 run passed all 9 bounded tests. A full
CPU execution of the `[128,2048]` fixture also emitted and replayed the exact
2.5-bpw object, passed the absolute source gate, and executed all nine
controls. Its result is recorded in `SOURCE_FREE_CPU_TARGET_RECEIPT.json`.
This constructed periodic fixture validates mechanics only. It is neither a
Qwen result nor a substitute for the pending CuPy receipt.
