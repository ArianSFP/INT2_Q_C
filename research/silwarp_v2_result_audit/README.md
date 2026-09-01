# SILWARP-v2 completed-run independent audit

Verdict: **PASS_AUTHENTICATED_HARD_KILL**.

The frozen auxiliary run is internally consistent and satisfies its
preregistered hard-kill rule at update 512. For both seeds, `U256 = U512 = 0`,
so `U512 < 0.10` and `U512 - U256 < 0.012` hold exactly. All 192 calibration
matrix evaluations selected the identity bypass, and all 56 jackknife folds
were zero. No SILWARP cell was promoted to confirmation.

## Sealed receipt

- Receipt: `audit_receipt.json`
- Full-file SHA-256:
  `0489c6d17dbe6be7c565319005c3bf3cd3c5acd02c0afc940ab719c8c8695b20`
- Canonical unsigned receipt SHA-256:
  `51e2e809e9221b850c9e169d88a6e4eb4fdf76d7c9f4e8eb945a5c73001d6ddd`
- Receipt verifier SHA-256:
  `c188bc4709226917debd6a2881cbcc7d9623d2a563651e3fd180f149b2076d51`
- Remote metadata verifier SHA-256:
  `b37b81bbd906db17408119e327244a44a3380e388b6faed3d0958d3dbd1f8c06`

The receipt's internal seal is SHA-256 over UTF-8 JSON serialized with sorted
keys, separators `(',', ':')`, `ensure_ascii=True`, and `allow_nan=False`,
after removing `seal.canonical_unsigned_sha256`. The sidecar
`audit_receipt.sha256` binds the exact pretty-printed file bytes.

## What was verified

- All eight frozen candidate hashes, including the GPU preflight receipt and
  launch sentinel.
- Exact 11-file output inventory, byte lengths, and SHA-256 hashes.
- Result, source-lock, sentinel, and both checkpoint canonical seals.
- Checkpoint 512's predecessor binding to both checkpoint-256 files; histories,
  runtime, source-lock, protocol-lock, and counter-randomness bindings.
- Both checkpoint ZIPs contain exactly 216 unique expected float32 state arrays
  with the expected shapes, without deserializing their numeric payloads.
- All six model files: full hashes, 4,096-byte headers, zero padding, and
  471,558-byte FP16 parameter-payload hashes.
- The 101-line JSONL log: 98 source loads (82 fit, 16 calibration), two
  checkpoints (256 and 512), one terminal event, and no confirmation or pinned
  event text.
- All result floats are finite; rate/read ledgers and the hard-kill arithmetic
  match the frozen executable protocol.

## Access boundary

The audit did not mutate the frozen candidate or run, open a Qwen tensor
payload, deserialize checkpoint numeric arrays, import CuPy, initialize CUDA,
or launch new training. Binary artifacts were streamed for SHA-256; only
existing JSON metadata, JSONL events, ZIP/NPY headers, and model headers were
parsed.

The source lock discloses authentication-only full-file hashing of two Gate
confirmation files before training. The completed run contains no confirmation
load, numeric confirmation score, or pinned-panel access. Artifact-only review
cannot exclude arbitrary unlogged/out-of-band access; it establishes that all
provided artifacts consistently record those firewalls as closed.

The GPU receipt is not referenced by the sentinel/checkpoints/result, and the
result does not contain the final-checkpoint or adjacent-log hash. The
independent receipt deliberately roots those hashes together after checking
their semantic consistency.

## Reproduction

On the RunPod, with the frozen paths still present:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  python3 -B verify_remote_metadata.py \
  --candidate /workspace/INT2__compression/silwarp_candidate_v2_c3f896b9 \
  --run /workspace/INT2__compression/silwarp_aux_run_v2_c3f896b9_v1 \
  --log /workspace/INT2__compression/silwarp_aux_run_v2_c3f896b9_v1.log
```

Verify the receipt itself with:

```sh
python3 -B verify_receipt.py audit_receipt.json
sha256sum -c audit_receipt.sha256
```
