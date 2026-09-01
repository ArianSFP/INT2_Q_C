# Independent RAVEL-6144-v1 result audit

Verdict: **PASS the narrow emitted HARD_KILL, with a result-only oracle replay
limitation.**

This audit carries exact copies of the sealed nine-file v1 source package and
the completed three-file RunPod result. It did not modify either origin and
opened no model payload, decoded panel, fresh validation data, CuPy runtime,
GPU, or network resource.

## Recomputed result

The 18 emitted per-matrix rows sum to the sealed baseline exactly:

| Quantity | Recomputed value |
|---|---:|
| Panel SSE | 500.39553685426534 |
| Panel energy | 16192.89450885593 |
| Holdout SSE | 166.91804981515884 |
| Holdout energy | 5279.494811461567 |
| Holdout F at 2.5 bpw | 1.011721345475929 |
| Finite fit-table SSE | 167.0011003655521 |
| Finite favorable F | 1.0187421104347276 |
| Source-leaking oracle SSE | 166.82239040215273 |
| Oracle capture | 0.0005730920838821207 |
| Oracle favorable F | 1.0176519417779304 |
| Oracle margin above 0.8 | 0.21765194177793035 |

The emitted oracle SSE is below both the zero-correction SSE and the finite
fit-table SSE by more than its declared numerical tolerance. Its favorable
`F` remains far above `0.8`, so the status
`HARD_KILL_RAVEL6144_V1` follows. Matched controls were correctly skipped by
the frozen stage-0 gate.

The source implementation uses the repaired raw-SSE sufficient statistics
`sum(scale*residual)` and `sum(scale**2)`, contains both dominance gates, uses
noncyclic self-clamped features, and consumes authenticated immutable input
snapshots. The result binds that exact runner and source manifest.

## Packet and completion findings

The independent parser—not the producer packet module—verified:

- exact 16,384-byte packet and 4,096-byte aligned header;
- canonical versioned semantics and one shared table;
- valid LF termination and all-zero header padding;
- exact table hash `e4418f...6734`;
- 6,144 finite FP16 values, 4,427 nonzero;
- value range `[-0.5126953125, 0.46923828125]`;
- packet/source/result bindings;
- canonical result and completion locks; and
- the completion marker’s exact member sizes and hashes.

The one-table side ledger is correct: `0.004629629629629629` bpw, leaving
`2.4953703703703702` bpw for the coarse payload. The conservative cold-read
amplification is `1.1805555555555556 < 2`.

## Limitation

The result does not emit the 6,144 per-cell weighted numerators and
denominators or the source-leaking FP64 oracle table. Therefore this result-only
audit can verify the correct sealed algorithm, recompute every emitted row/sum,
and independently prove the emitted dominance inequalities, but cannot
recompute the oracle SSE itself without reopening the forbidden payload. The
hard kill is consequently an integrity-checked, source-bound result rather than
an independent GPU/payload replay.

The claim remains narrow: this kills only the frozen one-table RAVEL-6144-v1
favorable family. It is not an achieved reduced-rate codec, activation-aware
evidence, or a converse for arbitrary universal SwiGLU-MoE residual coding.

## Replay

On a native Python 3.12 installation:

```bash
INT2_PROJECT_ROOT=/workspace/INT2__compression/INT2_Q_C
AUDIT_DIR="$INT2_PROJECT_ROOT/research/ravel_decoded_residual_lut_stage0_v1_independent_result_audit_20260901"
/usr/bin/python3.12 -B -I "$AUDIT_DIR/verify_audit.py" --audit-dir "$AUDIT_DIR"
cd "$AUDIT_DIR"
/usr/bin/python3.12 -B -I test_audit.py
```

Both commands operate only on the sealed audit closure and its copied evidence.
