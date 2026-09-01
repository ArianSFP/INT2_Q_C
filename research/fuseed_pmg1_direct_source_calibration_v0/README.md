# FUSEED-PMG1 direct source calibration v0

Work in progress.  This distinct source-free package hardens the only runtime
shape that survived the frozen exploratory timing gates: one complete u32
scan of the already-frozen `CURRENT_PMG_GATE_UP_DIRECT_BF16` ABI and one exact
FP64 source objective.  It reuses the original v1 ABI1 coordinate subset and
the original common/full/validation sets; it does not choose new coordinates
after observing performance.

No result in this directory authorizes Qwen/model access until the plan,
runtime/compiler, three-replay direct/sequential/Torch parity, exact full-shard
Top-K+journal timing, and an independent source audit all pass.

The first source-free launch, script SHA-256
`545719440e76ee9e5ab7aa17a20e222ddc8f4befaea8871ae11e885d0134e90b`,
failed before CUDA initialization because Python isolated mode correctly
ignored `PYTHONPATH`, so Torch was unavailable.  It created no result/output
directory.  The successor explicitly inserts only the already hash-bound
system package root after authenticating the raw environment and its runtime
files; the isolated `-I` launch remains mandatory.

The next launch, script SHA-256
`80f74cebfbe327717ea4d5e28c75678e156bef0dc6de93ec14404b22d9227fac`,
also stopped before CUDA initialization: prepending that root shadowed the
bound venv's CuPy/NumPy and tripped the exact version gate.  It likewise
created no output.  The current script appends the authenticated Torch root,
preserving the venv package precedence.
