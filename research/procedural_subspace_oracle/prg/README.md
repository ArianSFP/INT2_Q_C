# Procedural PRG union-of-subspaces screen

The complete method, results, exact source hashes, physical-rate/read ledger,
claim boundary, and reproduction commands are in [`RESULT.md`](RESULT.md).

Frozen confirmation:

- 18 Qwen matrices, 1,024 sampled vectors per matrix;
- raw and expert-affine XKLT representations;
- block dimensions 64, 128, and 256;
- Gaussian-QR, Rademacher-QR, and signed/permuted Hadamard bases;
- 16, 64, and 256 deterministic seed libraries;
- six-fold leave-one-expert-out hyperparameter selection;
- exact matched-energy Gaussian controls;
- 324 confirmation candidates.

The leakage-safe free-oracle ratio is `0.9979845528099698`, only 0.2015%
below matched Gaussian (`s=0.00145530487129292` bpw).  The optimistic charged
coefficient model gives `F=1.0535966954` at 2.5 bpw.  Both rates have exact
1.0x expert cold reads.  The family is hard-killed.

Key seals:

- algorithm:
  `6e40f954ae1ace27793082c9b435e80fb2ec99fbe5e773fba851dbae5fe52d56`
- confirmation file:
  `c027c9b15e6107bc3d0ab997775d90c42ccb2759561a5060cab88619aa2a1148`
- confirmation internal lock:
  `bc3391b286afb336175f39f585b2f28b87ca5911ae374e11985c3dd96b3af981`
- verification receipt lock:
  `ea62a950d490e949eae85c47b8a90c626734bceeab02f0f7a812ebdb9e6eee6a`
