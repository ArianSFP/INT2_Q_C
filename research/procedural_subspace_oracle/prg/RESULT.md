# Procedural union-of-subspaces oracle: confirmed early kill

## Outcome

This family is rejected.  On the pinned 18-matrix Qwen panel, even the
deliberately unimplementable free oracle recovers only a tiny matched
directional advantage:

| evaluation | source / matched Gaussian | percent below matched Gaussian | structural advantage `s=-0.5 log2(ratio)` |
|---|---:|---:|---:|
| panel-leaky best | 0.9853276077834638 | 1.4672392216536156% | 0.010662307218547315 bpw |
| six-fold leave-one-expert-out | 0.9979845528099698 | 0.20154471900302173% | 0.0014553048712929201 bpw |
| required | 0.8 | 20% | 0.16096404744368115 bpw |

The panel-leaky best is
`raw:d256:k252:hadamard:K16`.  It is already 15.10 times short of the
required structural advantage.  The leakage-safe result is 110.61 times
short.  The broad 96-vector/matrix smoke screen also covered nonlinear
power-law and sparse procedural atoms.  Its initially larger effect vanished
when expanded to 1,024 vectors per matrix, demonstrating why the confirmation
was required.

This is an early-kill result for the tested family, not a proof that every
conceivable procedural code is impossible.

## Tested construction

Each logical Qwen weight matrix was split into contiguous blocks with
`d in {64,128,256}`.  A seed selects a deterministic procedural basis.  The
screen covered:

- dense Gaussian QR bases;
- Rademacher/binary QR bases;
- signed square-root nonlinear PRG atoms followed by QR;
- 75%-sparse ternary PRG atoms followed by QR;
- signed/permuted Sylvester-Hadamard bases.

For low-dimensional candidates the decoder reconstructs in a `k`-dimensional
span.  For near-full candidates the equivalent excluded complement of rank
`d-k` is evaluated directly.  Coefficient counts span `k=1`, low rank,
half rank, and `d-8`/`d-4`.  The broad screen uses up to 64 seeds; the
confirmation uses 16, 64, and 256 seeds.

The free oracle is intentionally favorable:

1. Every block receives its exact norm for free.
2. Every seed receives continuous least-squares coefficients for free.
3. Every block chooses its best seed for free.
4. The procedural library and generator are charged zero table bits.

For each source block, the control is an independent iid Gaussian direction
rescaled to exactly the same block energy.  Source and control use the same
bases, continuous fitting, and best-seed selection.  Therefore
`D_source / D_control = 2^(-2s)` isolates the Qwen-specific directional
effect rather than generic vector-quantizer efficiency.

Both raw logical matrices and the frozen expert-affine XKLT representation
were evaluated.  Hyperparameter selection is repeated in six
leave-one-expert-out folds: the held-out expert never selects representation,
dimension, coefficient count, family, or seed count.

## Optimistic physical-rate accounting

The rate model is an intentionally favorable engineering comparator.  It
charges all of the following inside each private expert stream:

- one 512-bit expert-local header containing framing and generator metadata;
- `log2(K)` seed bits per block;
- 8 scale bits per block, while still granting the scale itself exact;
- all remaining physical bits to `k` coefficients;
- zero shared-table bits.

Coefficient quantization is modeled with the ideal Gaussian RD factor
`2^(-2q)`, where `q` is the exact remaining bits per coefficient.  This is
not a realized coefficient codec and is not an information-theoretic lower
bound for non-Gaussian coefficients.  The hard-kill decision rests on the
separate free-coefficient directional test above, as required by the protocol.

| requested rate | actual physical rate | best panel-leaky candidate | ideal coefficient bits | optimistic source MSE | `F=D*2^(2R)` | LOO `F` |
|---:|---:|---|---:|---:|---:|---:|
| 2.15 | 2.1500006781684027 | `raw:d256:k252:hadamard:K256` | 2.120525380291005 | 0.053486994151452055 | 1.0536044151150357 | 1.0536856727423871 |
| 2.5 | 2.5 | `raw:d256:k252:hadamard:K256` | 2.4760802469135803 | 0.03292068199676137 | 1.0534618238963638 | 1.0535966954112306 |

The target is `F <= 0.8`.  At 2.5 bpw, this oracle is 5.346% above the
Gaussian limit and 31.683% above the target `F`.  This establishes that the
tested Gaussian-RD coefficient model does not close the gap; it does not by
itself exclude a distinct non-Gaussian coefficient codec.

The expert stream is fully private and procedural decoding needs no weight
table from another expert.  The exact cold-read ledger is therefore:

| requested rate | physical bytes per expert | bytes read for one expert | cold read amplification |
|---:|---:|---:|---:|
| 2.15 | 1,268,122 | 1,268,122 | 1.0x |
| 2.5 | 1,474,560 | 1,474,560 | 1.0x |

Thus bandwidth is excellent, but distortion is decisively inadequate.

## Frozen provenance

- Plan lock SHA-256:
  `99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d`
- Plan file SHA-256:
  `8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868`
- Expert-affine header SHA-256:
  `3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538`
- Oracle source SHA-256:
  `6e40f954ae1ace27793082c9b435e80fb2ec99fbe5e773fba851dbae5fe52d56`
- Confirmation JSON SHA-256:
  `c027c9b15e6107bc3d0ab997775d90c42ccb2759561a5060cab88619aa2a1148`
- Confirmation internal lock:
  `bc3391b286afb336175f39f585b2f28b87ca5911ae374e11985c3dd96b3af981`
- Independent verification receipt lock:
  `ea62a950d490e949eae85c47b8a90c626734bceeab02f0f7a812ebdb9e6eee6a`

The independent verifier reopened the sealed plan, recomputed its canonical
lock, rehashed all 18 BF16 sources (56,623,104 bytes), checked the oracle
source hash, recomputed every candidate identity, every `F` identity, both
physical ledgers, and all 12 cross-fit selections.

### Exact source hashes

| ordinal | tensor | BF16 SHA-256 |
|---:|---|---|
| 0 | `model.layers.5.mlp.experts.18.gate_proj.weight` | `fe4fd2b8438d868a4b118df31f2886d36c2178c93132e5738e64008d1717a51c` |
| 1 | `model.layers.5.mlp.experts.18.up_proj.weight` | `857b57d1d37140bf10dbc582884c73c632f5a58cd1367f342c6903900e2b376b` |
| 2 | `model.layers.5.mlp.experts.18.down_proj.weight` | `8a1d32393816267ff6050d613a541250d73ff44a6ef6b2c43671d5904e1c7fe0` |
| 3 | `model.layers.12.mlp.experts.7.gate_proj.weight` | `eb56b4470fd98d169eba647262da183b66a94569a7a3a0869ac4d3148357abca` |
| 4 | `model.layers.12.mlp.experts.7.up_proj.weight` | `825cded5f7994df0363a4d170461964210ad56a8bfaecaf25b6166a6f5e1f156` |
| 5 | `model.layers.12.mlp.experts.7.down_proj.weight` | `160bbb9013002ff7301245402ff45654ac410f93208de8d42965b9b34b45df18` |
| 6 | `model.layers.18.mlp.experts.20.gate_proj.weight` | `74dbcaca0211e35a73cb18299fe3ff29bae0aeab58beb546f3c85d069b9fbaf6` |
| 7 | `model.layers.18.mlp.experts.20.up_proj.weight` | `8f44135aa4099014c740c99c87c037dad49b623a1e15bdceea1abe7917775b07` |
| 8 | `model.layers.18.mlp.experts.20.down_proj.weight` | `b0910cf461fbae04f904b4329c7149e7df68ed65caf6d8abb1b690966edad064` |
| 9 | `model.layers.28.mlp.experts.83.gate_proj.weight` | `236f15d3fea493b4b5012f3638d82b5e3db5429b952a93a81535d536d83e4867` |
| 10 | `model.layers.28.mlp.experts.83.up_proj.weight` | `95f654526d3de5860112020b170402e9d72c590437f4dd8518d15b1b07f9ff47` |
| 11 | `model.layers.28.mlp.experts.83.down_proj.weight` | `1b64f058bc1b7ceb755d2526b4196cd06c3879120f114ebbd13567e3971757df` |
| 12 | `model.layers.36.mlp.experts.76.gate_proj.weight` | `b77e0bd39b951ad983ffd511347f548cbda00806ab1dd300786df108cae1ad3f` |
| 13 | `model.layers.36.mlp.experts.76.up_proj.weight` | `b227044ca4db76a59086646727808b87f20664ab73e6ada19e106db38675fd26` |
| 14 | `model.layers.36.mlp.experts.76.down_proj.weight` | `442027369ca97e1ab072176a2a05a072329a36a90c62fc67c1946f1edeed4b69` |
| 15 | `model.layers.45.mlp.experts.41.gate_proj.weight` | `8c064b7b0fd4e04dd5c033b4264b2d3fe594ab95ab387d8a06d711e99a47eb9b` |
| 16 | `model.layers.45.mlp.experts.41.up_proj.weight` | `ba210f1528279f64241f3c01ba7cf5a722cdd0b10288448bc8b1c35b00dc4939` |
| 17 | `model.layers.45.mlp.experts.41.down_proj.weight` | `35de3de5ab4838b4777e3c0fe808b8faf3bd0fc24f934185a1a9800521362349` |

## Reproduction

The confirmation was run CPU-only on the supplied RunPod while the GPU encoder
continued independently:

```bash
cd /workspace/INT2__compression
OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 nice -n 10 \
  /workspace/int2-cupy-venv/bin/python \
  procedural_subspace_oracle/procedural_subspace_oracle.py \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --output procedural_subspace_oracle/confirmation_v1.json \
  --vectors-per-matrix 1024 \
  --dimensions 64,128,256 \
  --representations raw,xklt \
  --families gaussian_qr,rademacher_qr,hadamard \
  --max-seeds 256 \
  --seed-checkpoints 16,64,256
```

Independent verification, including reopening and rehashing the sources:

```bash
/workspace/int2-cupy-venv/bin/python \
  procedural_subspace_oracle/verify_procedural_subspace_oracle.py \
  procedural_subspace_oracle/confirmation_v1.json \
  --plan strata_expert_affine_milestone_v1/plan.lock.json \
  --algorithm procedural_subspace_oracle/procedural_subspace_oracle.py \
  --output procedural_subspace_oracle/confirmation_verification_receipt.json
```

No GPU library or CUDA API is imported by either program.
