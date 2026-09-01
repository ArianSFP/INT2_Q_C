# Independent result audit: FOSP auxiliary gross screen v0

Verdict: **PASS** for the exact frozen auxiliary gross-bound result and its
scoped hard kill. This is not validation, pinned-panel, legal-path/FP16,
finite-codec, or target-achievement evidence.

The audit bound the exact three-file producer closure, byte-exact frozen
oracle and bindings, all six permitted auxiliary tensors, and a new disjoint
RunPod replay. The replay result is structurally identical to the producer
result after deleting only `runtime.elapsed_seconds`; the replay file has
SHA-256 `7e44d40926d578c88083d9a40a77334c55ca70710a4e203cef566e839156e611`.

## Independent findings

An independent NumPy binary64 replay decodes the six BF16 files, rebuilds the
two `(768,3,2048)` source tensors, evaluates all `2*768*767` nonself
regressions through the held frozen oracle, and obtains the exact predecessor
hashes `fbd013a340582a1a060ab54b4022759e00ff0349b68a93e4aa0e8fc6f793a81d`
and `dcf7c88325cef5fdb6dbb1635798bbada01e2338325a898a8d4d494e18086ab6`.
Each score matrix contains exactly 768 negative-infinity entries, all on the
diagonal, and every selected predecessor is distinct from its target.

The gross relaxation contains every legal path: each legal path edge into a
target is one permitted nonself entry in that target's row and is at most the
row maximum. Summing the legal path's target subset cannot exceed the sum of
all 768 nonnegative row maxima used by the relaxation. The verifier also
exhaustively checks this inequality for every permutation at sizes 2 through
7 using exact rational synthetic scores.

From the two result rows, the verifier recomputes total energy
`5341.155115313352`, gross capture `106.02983322590737`, energy reduction
`0.01985147986470467`, gross `s=0.014463860060408446 bpw`, net side-adjusted
`s=-0.010379143954348498 bpw`, fraction of required gross
`0.0778434399926085`, and optimistic `F=1.0144925621732728`. It independently
rebuilds the complete physical ledgers at 2.15, 2.30, and 2.50 bpw.

The held runner compares the raw gross relaxed statistic directly with the
frozen required gross `0.1858070514584381 bpw`. Since it misses, the reported
`HARD_KILL_GROSS_QWEN_RELAXED_NECESSARY_BOUND` and `early_stop=true` follow.
Controls and legal-path/FP16 work occur only after gross survival and are
absent from this result; no control-corrected statistic can trigger this kill.
The access ledger is six auxiliary files and zero fresh-validation,
pinned-panel, and network operations.

## Verification

From this audit directory's parent on the authorized RunPod:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -B \
  free_order_swiglu_path_aux_gross_v0_independent_result_audit_20260901/verify_audit.py \
  --audit-manifest-sha256 <EXTERNAL_PIN_FROM_RELEASE_HANDOFF> \
  --workspace /workspace/INT2__compression
```

The verifier holds, hashes, and stability-checks every external evidence file
before use, strictly parses JSON, rejects non-regular named evidence, and
enforces exact producer and audit closure. It requires an explicit source
root and external audit-manifest pin.
The release handoff records that pin externally; embedding it in the receipt
would be circular because the manifest hashes the receipt.
