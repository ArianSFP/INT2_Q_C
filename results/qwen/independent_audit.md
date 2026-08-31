# Independent Audit: Qwen heldout32 exact_v2 vs rht_v2_postrms

Audit date: 2026-08-31
Canonical source: read-only audit of the RunPod workspace, with matching local
mirror hashes for the manifest, both summaries, both pack/unpack audits, both
panel reservoirs, and sampled decode JSONs. Connection details are intentionally
redacted from the public artifact.

## Headline result

- Exact energy-weighted relative MSE: `0.063198737741261`
- RHT energy-weighted relative MSE: `0.052894484749271`
- Absolute improvement from RHT: `0.010304252991990`
- Relative improvement from RHT: `16.304523%`
- Logical payload delta (`rht - exact`): `89406` bits
- Payload headroom delta (`rht - exact`): `-89406` bits
- Block win/loss count: `19` improved, `13` worsened
- Gate outcome: `exact` failed the joint rate/MSE gate; `rht` passed it.

## Manifest and implementation verification

- Manifest SHA-256: `3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55`
- Manifest revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Checkpoint: `Qwen/Qwen3-30B-A3B`; block count: `32`; block length: `262144`
- Same revision for both variants: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Shared decoder / map / packer / unpacker hashes: `2e1e484bf8ba98d493cfda55d4b23e275267e097e08907f5a9c606ae7350c797`, `a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef`, `c5fda34242153365dac07b5990bcd1fa19f0ac98d2512d47c3c8e1ec2a81dde8`, `cf7113c3fbc6340f0870dadcf7608739aa651f5706befa163b5d13516dac7e07`
- Encoder hash changed only where expected: `exact=95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8`, `rht=062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0`
- GPU/CuPy pins: `14.2.0` on `NVIDIA GeForce RTX 5090`
- Python entrypoint: `/workspace/int2-cupy-venv/bin/python`
- Manifest/order/source-hash alignment: `32/32` blocks matched in both variants.
- Independent decode status: `32/32` blocks passed in `exact` and `32/32` blocks passed in `rht`.
- RHT adapter check: `32` unique seeds, `side_bits=0` for every block, mode `hadamard_rademacher_splitmix64`.

## Reservoir and payload accounting

- `exact` panel SHA-256: `9388790c3cdbab5b9b33b676ced196090d81ba0422eb6fecfdd014bd2d054cf5`
  payload capacity `18030848` bits; logical payload `17916908` bits; headroom `113940` bits; physical panel `2254144` bytes
- `rht` panel SHA-256: `55d347c02ef1382ce209050d539f4e336dd7477125e4319e8b78d3067a436aac`
  payload capacity `18030848` bits; logical payload `18006314` bits; headroom `24534` bits; physical panel `2254144` bytes

## Per-role comparison

| Role | Exact MSE | RHT MSE | Improvement | Rel. % | Mean bits delta | Improved/Worsened blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_o | 0.132569248 | 0.052938480 | 0.079630767 | 60.067% | 20,356.00 | 1/1 |
| embedding | 0.082078330 | 0.052938448 | 0.029139882 | 35.503% | 8,171.50 | 1/1 |
| router | 0.077265604 | 0.052850155 | 0.024415449 | 31.599% | 11,268.00 | 2/0 |
| expert_gate | 0.052956181 | 0.052859494 | 0.000096687 | 0.183% | 226.50 | 5/1 |
| attention_k | 0.053002567 | 0.052948640 | 0.000053927 | 0.102% | 433.00 | 2/0 |
| expert_down | 0.052925121 | 0.052888039 | 0.000037082 | 0.070% | 182.50 | 4/2 |
| attention_q | 0.053064906 | 0.053028145 | 0.000036762 | 0.069% | 3,564.00 | 1/1 |
| expert_up | 0.052857170 | 0.052831516 | 0.000025654 | 0.049% | 36.33 | 3/3 |
| lm_head | 0.052865427 | 0.052956499 | -0.000091072 | -0.172% | -83.00 | 0/2 |
| attention_v | 0.052851712 | 0.052976376 | -0.000124664 | -0.236% | -342.50 | 0/2 |

## Most changed blocks

Top improvements:

| Block | Role | Improvement | Rel. % | Bits delta |
| --- | --- | ---: | ---: | ---: |
| attention_o.l47.b31 | attention_o | 0.146341104 | 73.471% | 40515 |
| embedding.b0 | embedding | 0.041991761 | 44.250% | 16264 |
| router.l0.b0 | router | 0.025423624 | 32.480% | 13325 |
| router.l47.b0 | router | 0.005199886 | 8.960% | 9211 |
| attention_q.l0.b0 | attention_q | 0.000650793 | 1.214% | 4662 |

Top regressions:

| Block | Role | Improvement | Rel. % | Bits delta |
| --- | --- | ---: | ---: | ---: |
| attention_o.l22.b2 | attention_o | -0.000194433 | -0.368% | 197 |
| expert_down.l47.e0.b0 | expert_down | -0.000177476 | -0.336% | -632 |
| embedding.b1186 | embedding | -0.000157977 | -0.299% | 79 |
| lm_head.b600 | lm_head | -0.000131338 | -0.249% | -262 |
| attention_v.l47.b3 | attention_v | -0.000131299 | -0.249% | -237 |

## Caveats

- The independent distortion recomputation uses the decoded JSON aggregation fields `fp16_sse_sum_fp64` and `source_energy_sum_fp64`. Those reproduce the published aggregates exactly, but this audit did not rebuild the decoder from source and re-parse raw payload bits independently of the shipped decode artifacts.
- Local and remote artifacts were matched by hash for the manifest, summaries, pack/unpack audits, panels, and sampled decode JSONs before using the remote canonical workspace as the computation source of truth.
- Full per-block decode statuses, per-block payload hashes, record hashes, source hashes, and paired deltas are in the companion JSON artifact.
