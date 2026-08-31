# Third-party notices

## Qwen3-30B-A3B

The real-weight evaluation uses selected BF16 ranges from
[`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B), pinned to
revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

The model card identifies the model license as Apache-2.0. A copy of those
terms is included at [`third_party/QWEN_LICENSE`](third_party/QWEN_LICENSE).
The repository does not redistribute the original BF16 source blocks or
safetensors shards. The small `.plrsv2` files in `results/` are transformed,
quantized derivatives of selected model blocks and retain their source-model
attribution. They are not a usable or complete Qwen checkpoint.

## Polar-lattice construction

The polar-lattice construction is based on:

- Ling Liu, Jinwen Shi, and Cong Ling, *Polar Lattices for Lossy
  Compression*, [arXiv:1501.05683](https://arxiv.org/abs/1501.05683).
- The authors' public MATLAB implementation,
  [`graceBaoXP/PolarLatticeQuantization`](https://github.com/graceBaoXP/PolarLatticeQuantization),
  pinned at commit `458187b9b03db1768a4b72d617e591f7862f6fca`.

The upstream repository did not expose an explicit license file at the pinned
commit when this package was prepared. Its source and large reliability-table
collection are therefore not vendored here. Reproduction instructions clone
the upstream repository separately, and users are responsible for reviewing
its terms. The Python files in this repository are an auditable reference
implementation/reproduction scaffold; this notice is attribution, not a
license grant for upstream material.

The codec requires a compact frozen decoder set map derived from the numerical
reliability tables in that upstream MATLAB repository. Because no upstream
license grant was visible, neither the tables nor the derived map are
redistributed here. `tools/build_decoder_map.py` generates the map locally
from a separately obtained, commit-pinned upstream checkout and accepts it
only if its SHA-256 is
`a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef`.
Users remain responsible for obtaining and using the upstream material under
appropriate terms.

## Repository licensing

No top-level software license was selected on behalf of the repository owner.
Until the owner adds one, normal copyright defaults apply to repository-native
code and documentation. Third-party components retain their own terms.
