# Permutation-aligned expert-template screen

Decision: **early kill**.

This CuPy screen tested a structural opportunity absent from the earlier
coordinate-aligned template experiment: exact SwiGLU-neuron permutation
alignment between two fixed, non-pinned layer-15 experts.  It deliberately
gave the candidate an exact uncharged reference expert, the globally optimal
768-by-768 assignment, and an exact uncharged least-squares coefficient for
every matched neuron.

The source pair captured only `0.0029652832` of joint Up/Down energy.  Two
moment-matched iid-Gaussian controls captured `0.0025901852` and
`0.0025677820`; subtracting the worse control leaves a Qwen-specific capture of
only `0.0003750979`.  Composing optimistically with the strongest existing
structural result would require `0.1456620755`.  The measured signal is about
`0.258%` of that requirement before charging the permutation, template,
coefficients, or framing, so a cross-fitted multi-expert codec is not
justified.

The result concerns one fixed auxiliary pair and is not a universal converse
for every permutation-aware model.  Its extremely favorable free-side upper
screen is nevertheless decisive for a single shared aligned template.

## Reproduction

On the supplied RunPod:

```bash
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python permutation_pair_screen.py \
  --source-dir /workspace/INT2__compression/qwen_weight_cache/rd_structure_diag_cross_expert \
  --output pair_result.json
```

Artifacts:

```text
dc9cc00dfbfb9e378291bb0abf67ff9d5c21d62a0f6c1f3c85df1b947d3dca00  permutation_pair_screen.py
ba22a5ac76a6cc697f63899787ab85396a5b00dc2d764299473ecb59e3a52a52  pair_result.json
```

The result binds all four BF16 auxiliary source hashes and records
`pinned_panel_opened=false`.
