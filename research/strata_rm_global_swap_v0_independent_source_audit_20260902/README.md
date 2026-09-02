# Independent source audit for `strata_rm_global_swap_v0`

This directory is an independently authored hostile audit of the immutable
producer root
`4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c`.
It does not modify the producer and must not touch payloads.

Run the CPU/source audit from a Python environment with NumPy:

```bash
python -I -B run_audit.py \
  --source /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v0 \
  --external-root /workspace/INT2__compression \
  --output /tmp/strata_rm_global_swap_v0_independent_audit.json
```

Run the separate real-CuPy parity audit in the pinned CuPy environment:

```bash
/workspace/int2-cupy-venv/bin/python -I -B run_real_cupy_audit.py \
  --source /workspace/INT2__compression/INT2_Q_C/research/strata_rm_global_swap_v0 \
  --external-root /workspace/INT2__compression \
  --output /tmp/strata_rm_global_swap_v0_real_cupy_audit.json
```

The launch controller must authenticate this audit directory before running
either command.  Both commands are source-free and confer no payload or
rate-distortion authority.
