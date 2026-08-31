# Third-party notices

## Qwen3-30B-A3B

The real-weight evaluation uses selected BF16 ranges from
[`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B), pinned to
revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

The model card identifies the model license as Apache-2.0. A copy of those
terms is included at [`third_party/QWEN_LICENSE`](third_party/QWEN_LICENSE).
The repository does not redistribute the original BF16 source blocks or
safetensors shards. The small `.plrsv2` files in `results/` and the
STRATA-XKLT-SC v2 `.bin` container are transformed, quantized derivatives of
selected model weights and retain their source-model attribution. They are not
a usable or complete Qwen checkpoint. Original BF16 payloads, reconstruction
arrays, safetensors shards, and access credentials are not included.

## Polar-lattice construction

The polar-lattice construction is based on:

- Ling Liu, Jinwen Shi, and Cong Ling, *Polar Lattices for Lossy
  Compression*, [arXiv:1501.05683](https://arxiv.org/abs/1501.05683).
- The authors' public MATLAB implementation,
  [`graceBaoXP/PolarLatticeQuantization`](https://github.com/graceBaoXP/PolarLatticeQuantization),
  pinned at commit `458187b9b03db1768a4b72d617e591f7862f6fca`.

The upstream repository did not expose an explicit license file at the pinned
commit when this package was prepared. Its original source and large
reliability-table collection are therefore not vendored here. Reproduction
instructions clone the upstream repository separately, and users are
responsible for reviewing its terms. Some Python reference-code comments
describe routines as direct translations of the named MATLAB implementation.
The absence of an upstream license grant is unresolved; publication here is
for byte-level research auditability and does not purport to grant reuse rights
in upstream-derived material. This notice is attribution, not a license grant.

Three files already present in the historical POLARIS release explicitly
describe vectorized translations or a direct port of the upstream routines:

- `src/polaris_sc_v2_encoder.py`
- `src/polaris_sc_v2_rht_encoder.py`
- `frozen/gaussian_confirmation/agent_root_polar_lattice_gate.py`

They remain historical evidence in the pre-existing repository and carry the
same unresolved reuse status. They are outside the new STRATA release
manifest. Their presence must not be read as a representation that upstream
redistribution permission exists.

The codec requires a compact frozen decoder set map derived from the numerical
reliability tables in that upstream MATLAB repository. Because no upstream
license grant was visible, neither the tables nor the derived map are
redistributed here. `tools/build_decoder_map.py` generates the map locally
from a separately obtained, commit-pinned upstream checkout and accepts it
only if its SHA-256 is
`a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef`.
Users remain responsible for obtaining and using the upstream material under
appropriate terms.

STRATA-XKLT-SC v2 does not use that external decoder map. It constructs its
six frozen sets procedurally from an unsigned-Q31 binary-erasure-channel
surrogate. The historical freeze and runtime receipts bind the base encoder as
`agent_polaris_qwen_rht_encoder.py`, SHA-256
`062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0`.
That file is withheld from this repository because its own comments identify
a direct port of the unlicensed upstream MATLAB implementation. The compact
verifier checks the retained hash/path bindings without redistributing it.

## Repository licensing

No top-level software license was selected on behalf of the repository owner.
Until the owner adds one, normal copyright defaults apply to repository-native
code and documentation. Third-party components retain their own terms.
