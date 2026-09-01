# TACTIC-DH384 v2 synthetic CuPy preflight

This directory records the root-reviewed, synthetic-only RunPod preflight for
[`../tactic_conditional_dyadic_coset_v2`](../tactic_conditional_dyadic_coset_v2/README.md).
It did not accept a model path, coarse lock, source matrix, reconstruction, or
quantizer payload.

On the RTX 5090 with CuPy 14.2.0, the CPU and GPU implementations agreed on
the fixture energy exactly and on projected energy to floating-point roundoff:

```text
energy:            1378.1384775399747 (CPU and GPU)
projected energy:  127.20384966527631 (CPU)
                   127.20384966527632 (GPU)
```

The source manifest was
`f8de593784638cf7719d08ddda7061f4912166021214fb7a2894862a53050662`.
The emitted selector packet is 16,384 bytes with SHA-256
`0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad`.
The `receipt.json` file SHA-256 is
`c91c702a603848abf0b7bfdac0c8e01740bf8cd4d80baa93b0243647fc2c043f`;
its embedded canonical body seal is
`fd3b18bd3ca7f725457509c1d6880ea6a95b057334eed8945e6f0157ee25d54a`.

This proves only synthetic CPU/GPU parity and ledger arithmetic.  It is not an
independent source audit, a lower-rate coarse artifact, a Qwen result, or a
compression claim.
