# TACTIC actual coarse N18 v6

Date: 2026-09-02

Status: frozen source-only successor awaiting an external source-free CuPy
smoke and an independent result audit. No RunPod, CUDA, Qwen/model payload or
network was accessed while building this sibling. The smoke has **not** been
run. V4 and v5 remain unchanged.

V6 preserves the literal `TACN18C4` reservoir: 78,592 bytes per 262,144-value
full record, exactly `307/128 = 2.3984375` bpw. It repairs six production
blockers without changing the packet grammar.

## Frozen repairs

1. The inherited ordered role ABI is exactly `gate`, `up`,
   `down_transposed`. `down` is never accepted or emitted. Transposition is an
   authenticated upstream input obligation.
2. Both executable entry points require CPython `-I -B`. They authenticate
   `source_auth.py` from an externally pinned manifest row, dynamically execute
   only authenticated sibling bytes, and retain no-follow descriptors for the
   complete source package through terminal verification.
3. The source-free smoke receipt is stored outside the immutable package. It
   binds the exact source manifest, source root, every source member, the
   `synthetic_cupy_smoke.py` executing inode, predecessor/runtime locks, an
   internal canonical-JSON seal and an external literal-file SHA-256.
4. The inverse installed in the inherited decoder is exercised with all
   indices 63. Its exact unnormalised N18 maximum is
   `32*262144 = 8,388,608 > 32767`. I32 is checked before facade retention;
   no copy/downcast is allowed; the same retained values then enter the F64
   reconstruction expression.
5. Traffic is reported in four disjoint ledgers: external compressed-file
   reads, host-memory parse/integrity lower bounds, scratch lower bounds and
   accelerator HBM. External accounting obeys
   `total = first_pass + reread`. Prebuffered encoder output has zero external
   reads; separately modelled one- and two-file-pass cases report one first
   pass and zero/one reread respectively. Host-memory work is not external I/O,
   and HBM is explicitly unmeasured, so this package has no `<2x` inference-HBM
   claim authority.
6. A result is written into a private staging directory with a non-authorizing
   pending completion record, fsynced, and published only with Linux
   `renameat2(RENAME_NOREPLACE)`. The staging FD remains open through an
   independent final-name reopen; every ordinary member and the pending record
   is rehashed and name/inode rebound. Only then is the pending record renamed
   no-replace to terminal `COMPLETE.json`. Existing targets are never replaced.
   A namespace interrupted before that terminal rename is not a result.
   Outputs inside the immutable source package are rejected before runtime
   work.

## Exact rate and scope

For the `[768,2048]` shape, each role has six full N18 records, so one expert
has 18 records, 1,414,656 coarse bytes and exact aggregate rate `307/128`.
This is only a shape fact. A bound run does not infer model, layer or expert
identity.

For arbitrary SwiGLU shapes, v4-compatible implicit-zero tail records remain
decodable but the aggregate rate is strictly above `307/128`. V6 labels those
as nonpromoting compatibility tails. It does not claim a universal arbitrary-
shape `<=2.5`-bpw result. A Qwen pilot and a universal-tail result are separate
scientific claims and require separate sealed audits.

## Source-only verification

After freeze, run only the standard-library checks:

```bash
python -I -B research/tactic_actual_coarse_n18_v6/verify_source.py
python -I -B research/tactic_actual_coarse_n18_v6/test_source_only.py
```

These checks do not initialize CUDA or touch a model payload.

## External source-free CuPy smoke

Only after the source manifest hash is independently captured, run:

```bash
python -I -B research/tactic_actual_coarse_n18_v6/synthetic_cupy_smoke.py \
  --repo-root /workspace/INT2_Q_C \
  --package-dir /workspace/INT2_Q_C/research/tactic_actual_coarse_n18_v6 \
  --package-manifest-sha256 <SOURCE_MANIFEST_SHA256> \
  --predecessor-lock-sha256 <PREDECESSOR_LOCK_SHA256> \
  --runtime-lock-sha256 <RUNTIME_LOCK_SHA256> \
  --receipt-output /tmp/tactic_actual_coarse_n18_v6_smoke.json
```

The receipt path must be absent and outside this package. The smoke uses a
deterministic synthetic BF16 tile plus a three-role zero frame. It does not
discover, enumerate or access model paths and is not a Qwen result.

## Bound pilot input ABI

The dispatcher accepts only this identity-free grammar:

```json
{
  "schema": "tactic-actual-coarse-n18-v6-input-manifest-v1",
  "geometry": {"intermediate": 768, "hidden": 2048},
  "roles": [
    {"role": "gate", "relative_path": "gate.bf16", "bytes": 3145728, "sha256": "..."},
    {"role": "up", "relative_path": "up.bf16", "bytes": 3145728, "sha256": "..."},
    {"role": "down_transposed", "relative_path": "down_t.bf16", "bytes": 3145728, "sha256": "..."}
  ],
  "output_directory_name": "coarse_result"
}
```

Model/checkpoint/layer/expert/tensor identity fields are forbidden. A launch
must pass both the external smoke receipt file hash and all frozen source/
runtime/input hashes. Its status remains nonpromoting until an independent
auditor verifies `COARSE.bin`, all receipts, `RESULT.json` and terminal
`COMPLETE.json`.
