# TACTIC-DH384 finite v3 external source-free CuPy smoke

The first frozen source (`cf5740c0...`) passed 25 standard-library tests but
failed safely on CuPy 14.2 because that version does not implement the `axis`
argument to `packbits`. No payload was opened. Its failure is preserved in the
finite package's `SUPERSESSION.md`.

The repaired source freeze is:

- manifest SHA-256:
  `bf0659d1fd6742768d14790ea980aa17321818d15e19ddd7d0dfaa8a223009b8`
- source root:
  `725991e0c1e10c67db4ba36097f80e78ffed158ea36b1b746bbbd6cef50ffa98`

On the RTX 5090 with NumPy 2.5.2 and CuPy 14.2.0:

- all 64 synthetic 48-byte records re-encoded independently;
- encoder and independent decoder corrections were bitwise numerically equal
  (`max abs difference = 0`);
- transform energy matched exactly in the reported FP64 reductions;
- transformed coefficient tail outside rank 376 was exactly zero;
- forced local and exhaustive same-codebook decisions had zero mismatches;
- finite SSE fell from `22.690393703204023` to `21.384562617226948` on the
  synthetic fixture.

This smoke initialized CuPy only after the source freeze. It accessed no Qwen
or model payload and no live v6 result. It is implementation parity evidence,
not a Qwen MSE result or launch authorization.
