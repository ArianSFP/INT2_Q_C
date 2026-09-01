# PSNO-v1 authenticated Qwen sparse-normal gate

This package executes the source-only proposal in
`../polar_sparse_normal_oracle_v1` on exactly the previously authenticated
18-matrix Qwen3-30B-A3B panel.  It neither requests nor reads fresh validation
data.  The prior NumPy derivation is imported only after its bytes are
authenticated, and every rebuilt polar normal must reproduce the prior binary64
SHA-256 exactly before CuPy sees it.

The hard ledger is deliberately source-leaky: an arbitrary selected normal
coefficient is reconstructed as its exact continuous value while paying only
one bit.  Support pays the exact enumerative length
`ceil(log2(binomial(n,k)))`; headers and mode labels are free.  Failure under
this ledger is therefore an early kill, not a finite-codec measurement.

For every discrete support option the runner analytically minimizes the
remaining ideal Gaussian payload term, then maximizes a valid Lagrangian lower
bound over a deterministic multiplier scan.  Arbitrary coordinates use the
exact best-k curve.  Fixed tile and diagonal-offset prefixes are measured
directly.  A separate gross relaxation gives each group count the capture and
DOF removal of the largest possible arbitrary coordinate set but charges the
smallest possible value set.  Only that relaxed curve may kill the broader
arbitrary-subset group family.

Eight Gaussian-rank controls preserve every absolute orthonormal coefficient
and exact per-matrix normal energy while randomizing spatial placement.  They
are diagnostics only; no corrected control statistic participates in a kill.

Run on the authorized RunPod from the repository root:

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/polar_sparse_normal_oracle_v1_qwen_gate_20260901/run_gate.py \
  --repo-root /workspace/INT2__compression/INT2_Q_C \
  --source-root /workspace/INT2__compression \
  --output INT2_Q_C/research/polar_sparse_normal_oracle_v1_qwen_gate_20260901/result.json
```

Then verify the result seal and all immutable lineage receipts:

```bash
/workspace/int2-cupy-venv/bin/python \
  INT2_Q_C/research/polar_sparse_normal_oracle_v1_qwen_gate_20260901/verify_result.py \
  --repo-root /workspace/INT2__compression/INT2_Q_C \
  --source-root /workspace/INT2__compression \
  --result INT2_Q_C/research/polar_sparse_normal_oracle_v1_qwen_gate_20260901/result.json \
  --receipt INT2_Q_C/research/polar_sparse_normal_oracle_v1_qwen_gate_20260901/verification_receipt.json
```

`result.json`, `verification_receipt.json`, and `RESULT_MANIFEST.json` are
generated artifacts.  No finite codec, achieved MSE, production kernel, or
fresh-data generalization claim is made.
