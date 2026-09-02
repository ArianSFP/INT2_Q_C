# V6 implementation map

| Frozen contract | Owning source |
|---|---|
| Exact retained v6 source closure and executing-entry binding | `source_auth.py` |
| Immutable full v4 and numerical dependency pins | `PREDECESSOR_LOCK.json`, `runtime_closure.py` |
| Exact isolated RTX runtime pin | `RUNTIME_LOCK.json`, `runtime_closure.py` |
| Literal `gate/up/down_transposed` ABI | `dispatcher.authenticate_inputs`, `successor_codec.encode_expert_frame_from_bf16_v6` |
| Installed I32 inverse and exact worst-case bound | `runtime_closure.load_runtime`, `synthetic_cupy_smoke.py` |
| No-copy I32 facade gate before reconstruction cast | `successor_codec.retain_canonical_symbols_i32`, `decode_tile_v6` |
| Exact aggregate re-encode and original-byte scoring | `successor_codec.decode_expert_frame_bytes_v6` |
| Exact `307/128` aggregate rational and nonpromoting tails | `successor_codec.exact_rate_record`, `dispatcher.run` |
| External/host/scratch/HBM traffic separation | `successor_codec.frame_ledger_v6`, decoder and result receipts |
| Source/self-bound external smoke receipt | `synthetic_cupy_smoke.py`, `smoke_contract.py` |
| Race-free absent-target result publication | `dispatcher.publish_atomic` |
| Hostile source-only checks | `test_source_only.py`, `verify_source.py` |

V6 does not alter v4 or v5. It preserves the literal v4 packet language and
executes manifest-authenticated source bytes under isolated mode. The one
numerical override is installed before any reservoir decode: the integer
inverse-Hadamard output remains little-endian I32 instead of encountering the
v4 I16 terminal gate.

The external source-free smoke is a prerequisite, not a checked-in result.
Its PASS can authorize a separately input-bound pilot launch; it cannot itself
establish Qwen MSE, universal tail rate, fine-code gain or inference-HBM
traffic. Any pilot output remains nonpromoting until an independent result
audit authenticates its literal publication members.
