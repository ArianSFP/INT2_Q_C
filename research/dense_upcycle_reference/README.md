# Dense-to-MoE upcycle-reference screen

Decision: **early kill**.

Qwen3-1.7B-Base and Qwen3-30B-A3B both use hidden size 2,048, and the dense
model's FFN width 6,144 equals `8 * 768`, the total intermediate width activated
by eight MoE experts.  Earlier Qwen MoE work also used dense-to-MoE
initialization.  This experiment therefore tested whether the public dense
checkpoint contains useful neuron-level ancestors of a Qwen3 MoE expert.

The screen was intentionally much stronger than a realizable codec.  For one
fixed, non-pinned layer-15 expert it provided exact uncompressed Up/Down tensors
from Qwen3-1.7B-Base layers 9 and 15 for free, selected the better layer for
free, solved the optimal one-to-one assignment of 768 target neurons into
6,144 reference neurons, and fitted separate exact FP64 scale and offset for
both roles and every neuron.  No reference, index, coefficient, or read cost
was charged.

The best raw capture was only `0.0051296784`.  The maximum of four matched
coordinate-scramble controls was `0.0050032978`, leaving just
`0.0001263806` Qwen-specific capture.  Optimistic composition with the best
existing structural result requires `0.1456620755`.  The apparent reference
gain is therefore chance-level and about 1,153 times too small; downloading or
searching the other 26 dense layers is not justified.

This rejects the tested public Qwen3-1.7B neuron-ancestor hypothesis, not every
possible private pretraining checkpoint or training lineage.

## Locked public ranges

Only four 25,165,824-byte BF16 tensor ranges were downloaded from the immutable
Qwen3-1.7B-Base revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.  The safetensors header, exact
HTTP ranges, shapes, and tensor hashes are sealed in
`dense_reference_manifest.json`; the 100 MiB of negative reference data is not
duplicated in git.

## Reproduction

```bash
python fetch_locked_dense_ranges.py --output-dir qwen3_1_7b_ranges

PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python dense_upcycle_pair_screen.py \
  --target-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --reference-dir qwen3_1_7b_ranges \
  --output dense_upcycle_pair_result.json
```

Artifact identities:

```text
c771242918ebfc985c52eb1d5495e3390ff4ae97a58c84d63a1458f5f1dadc5e  fetch_locked_dense_ranges.py
416457ccbd1aeafa3017a9feb4c65f8c7a187041c5d6ae230754e5c74279a7e9  dense_upcycle_pair_screen.py
f535a729750432dcc2cf5c95c78b32aaa32701f5b99d4ab4f3e63114cef294f6  dense_reference_manifest.json
36e7eb51f3eef51f88e6b08c562905c0e2797949b18327c35e2033363cd5db71  dense_upcycle_pair_result.json
```
