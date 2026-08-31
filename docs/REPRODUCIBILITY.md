# Reproducing POLARIS-SC-v2

This guide reproduces the published matched-Gaussian confirmation and the
paired 32-block Qwen evaluation. It is written for a fresh Linux checkout and
does not require, contain, or redistribute a Qwen checkpoint. The Qwen source
materializer downloads only the immutable byte ranges named by the frozen
manifest and rejects any block whose SHA-256 differs.

There are three useful verification levels:

1. **Artifact verification** validates the two checked-in fixed reservoirs and
   requires neither Qwen weights nor a GPU.
2. **Qwen reproduction** fetches 32 BF16 blocks, executes both the exact and
   RHT arms, packs and independently unpacks them, performs fresh-process
   decoding, and recomputes the paired and distribution audits.
3. **Gaussian reproduction** reruns the preregistered synthetic confirmation
   from a copy of its frozen workspace.

All commands below are run from the repository root.

The reproduction-facing entry points are:

```text
tools/build_decoder_map.py        locally derive/hash-check the required set map
tools/materialize_qwen_panel.py   fetch and verify all manifest source ranges
tools/run_qwen_release.py         map the published repository layout to the runner
tools/run_qwen_panel.py           hash-pinned encode/pack/unpack/decode engine
tools/audit_qwen_paired.py        same-source exact-versus-RHT audit
tools/audit_qwen_distribution.py  CuPy distribution and kurtosis audit
tools/verify_release.py            offline hashes/seeds/reservoir/result audit
tools/verify_gaussian_confirmation.py safe pristine-copy Gaussian runner
frozen/gaussian_confirmation/     original-basename Gaussian rerun workspace
```

`run_qwen_release.py` is the recommended convenience wrapper. The underlying
`run_qwen_panel.py` remains the normative, hash-pinned execution engine.

## Frozen identities

The Qwen evaluation is bound to:

- checkpoint `Qwen/Qwen3-30B-A3B`;
- revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`;
- manifest SHA-256
  `3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55`;
- 32 blocks of `N = 262144` values, or 8,388,608 exact BF16 source values;
- the official PolarLatticeQuantization repository at commit
  `458187b9b03db1768a4b72d617e591f7862f6fca`.

The block panel covers six blocks from each expert projection (gate, up, and
down), two from each attention projection (q, k, v, and o), and two each from
the embedding, LM-head, and router roles. It is a deterministic coverage
panel, not a probability sample.

## Reference environment

The published run used this environment:

| Component | Frozen value |
|---|---|
| Python | 3.12.3, GCC 13.3.0 |
| Operating system | Linux 6.8, x86-64, glibc 2.39 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| CuPy package | `cupy-cuda12x==14.2.0` |
| CuPy CUDA runtime | 12.9 |
| NVIDIA driver/runtime API | 580.126.09 / CUDA 13.0 |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Parallel workers | 4 |

The complete reference `pip freeze` is the four-package set pinned in
`requirements.txt`; `cuda-pathfinder==1.8.0` is CuPy's transitive runtime
locator dependency.
Another CUDA GPU may produce the same result, but bit-identical reservoir
hashes are asserted only for the pinned environment above.

Create the environment and fetch the pinned upstream construction:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p third_party
git clone https://github.com/graceBaoXP/PolarLatticeQuantization.git \
  third_party/PolarLatticeQuantization
git -C third_party/PolarLatticeQuantization checkout --detach \
  458187b9b03db1768a4b72d617e591f7862f6fca
test "$(git -C third_party/PolarLatticeQuantization rev-parse HEAD)" = \
  "458187b9b03db1768a4b72d617e591f7862f6fca"

python tools/build_decoder_map.py \
  --polar-repo "$PWD/third_party/PolarLatticeQuantization" \
  --output "$PWD/codec_data/polaris_sc_v1_decoder_map.npz"
```

The encoder reads the three upstream `n_18` reliability tables named
`Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_Lvl_{1,2,3}_n_18.mat`.
Do not substitute tables from another commit. The tables and their derived
decoder map are not redistributed because the pinned upstream repository has
no explicit license. The builder requires the exact commit and accepts only a
200,184-byte map with SHA-256
`a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef`.

Check the numerical runtime before spending time on an encode:

```bash
python - <<'PY'
import cupy as cp
import numpy as np
import scipy

print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("CUDA driver:", cp.cuda.runtime.driverGetVersion())
print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
PY
```

## Verify the checked-in reservoirs first

The compact artifact check exercises the strict standalone parser. It checks
the header, mixed-endian directory, every logical prefix, record-local tail
bits, fixed payload capacity, and the all-zero global suffix. Use new output
directories because the unpacker deliberately refuses to overwrite one.

Start with the dependency-free aggregate check. The generated decoder map is
optional for this structural verification:

```bash
python tools/verify_release.py
```

```bash
python src/reservoir_unpack_v2.py \
  --input results/qwen/exact_panel.plrsv2 \
  --output-dir /tmp/polaris-exact-unpacked \
  --audit /tmp/polaris-exact-unpack.json

python src/reservoir_unpack_v2.py \
  --input results/qwen/rht_panel.plrsv2 \
  --output-dir /tmp/polaris-rht-unpacked \
  --audit /tmp/polaris-rht-unpack.json

sha256sum results/qwen/exact_panel.plrsv2 results/qwen/rht_panel.plrsv2
```

Expected reservoir hashes are:

```text
9388790c3cdbab5b9b33b676ced196090d81ba0422eb6fecfdd014bd2d054cf5  results/qwen/exact_panel.plrsv2
55d347c02ef1382ce209050d539f4e336dd7477125e4319e8b78d3067a436aac  results/qwen/rht_panel.plrsv2
```

Each file must be exactly 2,254,144 bytes. This verification proves the
published reservoir structure and hashes; it cannot recompute weight-domain
MSE without the original BF16 source blocks.

## Materialize the Qwen source panel

The repository intentionally excludes all Qwen weights. The source tree is
ignored by Git. Review the Qwen license and upstream terms before fetching.

`tools/fetch_qwen_block.py` uses safetensors metadata and HTTP Range requests.
It downloads exactly one 524,288-byte, C-order BF16 block per invocation and
refuses a server response that does not honor the range. The wrapper below
materializes all 32 manifest entries into their exact manifest paths, checks
every SHA-256, and is safe to rerun when existing files are already valid:

```bash
mkdir -p repro
PYTHON_BIN="$PWD/.venv/bin/python"

"$PYTHON_BIN" tools/materialize_qwen_panel.py \
  --workspace "$PWD" \
  --manifest "$PWD/results/qwen/manifest.json" \
  --fetcher "$PWD/tools/fetch_qwen_block.py" \
  --python "$PYTHON_BIN" \
  --workers 2 \
  --audit "$PWD/repro/source_materialization_audit.json"
```

This creates 16 MiB of raw BF16 block data under
`qwen_polaris_heldout32/sources/`, plus small provenance sidecars. The fetcher
is pinned internally to the checkpoint and revision above. The panel runner
hashes every source again before encoding, immediately after encoding, and
during aggregation. A wrong or mutated source is a hard failure; it is never
silently redownloaded or replaced during an experiment.

### Source representation

Every tensor is flattened in safetensors/C row-major order without a
transpose. A block is the contiguous interval
`[canonical_block_index*N, (canonical_block_index+1)*N)`. BF16 is decoded
exactly by interpreting little-endian `u16`, shifting it into the high half of
`u32`, viewing that as binary32, and widening to binary64. There is no
centering, clipping, fitted offset, calibration data, QAT, or retry.

The extracted files each contain one block, so their
`source_local_block_index` is zero even when their canonical tensor-local
index is nonzero. Do not replace the local index with the canonical index when
invoking an encoder on these extracted files.

## Deterministic seeds and the RHT contract

Both seeds are derived from the canonical identity, not stored as an
uncharged per-block payload. For a manifest row:

```python
material = f"{revision}:{tensor}:{canonical_block_index}".encode("utf-8")
digest = hashlib.sha256(material).digest()
sc_seed_u32 = int.from_bytes(digest[0:4], "big")
rht_seed_u64 = int.from_bytes(digest[4:12], "big")
```

The runner independently rederives these values, rejects collisions in the SC
seeds, and rejects any seed or source identity mismatch.

For RHT block element `i`, SplitMix64 is evaluated at `rht_seed_u64 + i`.
The sign is `+1` when the mixed value's low bit is zero and `-1` otherwise.
With the unnormalized Walsh-Hadamard matrix `H_N`, the forward and inverse are

```text
y    = H_N diag(signs) x / sqrt(N)
xhat = diag(signs) H_N yhat / sqrt(N).
```

The transform is orthonormal and implemented in CuPy. It preserves energy and
requires no stored sign vector. The exact control skips this transform; all
other codec settings are identical.

## Codec and serialized metric

Both arms use the following frozen settings:

- six-level Construction-D polar lattice;
- block length `262144`;
- source sigma `1.0` after per-block RMS normalization;
- Gaussian test-channel distortion `0.05110`;
- 64 reconstruction levels at spacing `eta = 0.25`;
- randomized successive-cancellation decisions;
- one attempt per source block.

The encoder first writes a staging record:

```text
u32le logical_bit_length | f32le reconstruction_scale | padded payload
```

The global packer converts the scale through the normative chain
`RMS_FP64 -> binary32 RNE -> binary16 RNE` and emits:

```text
96-byte reservoir header
32 directory entries: u32be logical_bit_length | raw f16le scale
32 * 563464 fixed-capacity payload bits
```

Logical payloads are concatenated MSB-first. Only the global condition
`sum(lengths) <= block_count * 563464` matters; unused capacity is physically
zero-filled. The standalone unpacker extracts records in the independent
layout

```text
u32le logical_bit_length | raw f16le scale | locally padded payload.
```

The Qwen decoder uses the raw two FP16 bytes extracted from the reservoir. It
does not use the encoder's FP32 scale for the final metric. For block `b`, the
reported aggregate is

```text
energy-weighted relative MSE = sum_b ||x_b - xhat_b||^2 / sum_b ||x_b||^2.
```

It is not the unweighted mean of per-block ratios. For the RHT arm, the error
is measured after the independently implemented inverse RHT in weight space.

## Run the paired Qwen experiment

Use fresh, non-existing work directories. Each run writes an irreversible
execution lock before opening results and deliberately disables `--resume`.
If a run is interrupted, retain it for diagnosis and choose a new work
directory; do not remove its lock and relabel a partial result.

The exact control is expected to finish successfully but return status `2`
because it fails the preregistered MSE gate. The following shell fragment
distinguishes that expected scientific failure from an execution failure:

```bash
set +e
"$PYTHON_BIN" tools/run_qwen_release.py \
  --variant exact \
  --workdir "$PWD/repro/qwen_exact" \
  --polar-repo "$PWD/third_party/PolarLatticeQuantization" \
  --workers 4 \
  --python "$PYTHON_BIN"
EXACT_STATUS=$?
set -e
test "$EXACT_STATUS" -eq 2
```

Now run the RHT arm. It must return status zero:

```bash
"$PYTHON_BIN" tools/run_qwen_release.py \
  --variant rht \
  --workdir "$PWD/repro/qwen_rht" \
  --polar-repo "$PWD/third_party/PolarLatticeQuantization" \
  --workers 4 \
  --python "$PYTHON_BIN"
```

The wrapper supplies the frozen manifest, exact/RHT encoders, packer,
unpacker, real-source decoder, and decoder-map paths shown in the hash table
below. It invokes `run_qwen_panel.py` as a subprocess and preserves its exit
status. Pass `--manifest` only when auditing the byte-identical published
manifest at another location; the runner will reject a different hash.

On the reference RTX 5090, each arm took about 15.7 minutes with four workers
(`939.8 s` exact and `943.3 s` RHT). Each block is encoded once. There is no
seed search, result-dependent retry, or post-result parameter selection.

The real-source decoder imports the frozen core under its historical module
name. Therefore `src/agent_polaris_independent_decoder_v1.py` is an intentional
byte-identical compatibility copy of `src/independent_decoder_v1.py`; both
must retain SHA-256
`0652521fae5c77567e67cc9434adfb84b1a3cab53e5a79f250a3093a467be072`.

## Recompute the audits

The paired audit consumes the full generated work trees, including all 64
fresh decode JSON files. It cannot be run from only the compact checked-in
summaries.

```bash
"$PYTHON_BIN" tools/audit_qwen_paired.py \
  --manifest "$PWD/results/qwen/manifest.json" \
  --exact-dir "$PWD/repro/qwen_exact" \
  --rht-dir "$PWD/repro/qwen_rht" \
  --output "$PWD/repro/paired_audit.json"

"$PYTHON_BIN" tools/audit_qwen_distribution.py \
  --workspace "$PWD" \
  --manifest "$PWD/results/qwen/manifest.json" \
  --exact-summary "$PWD/repro/qwen_exact/summary.json" \
  --rht-summary "$PWD/repro/qwen_rht/summary.json" \
  --output "$PWD/repro/distribution_audit.json"
```

The second audit uses CuPy to reload every source block, recompute the RHT,
and measure raw and transformed kurtosis and tail energy. The separately
checked-in `results/qwen/independent_audit.{json,md}` records an additional
cross-host audit of the publication artifacts.

### Expected Qwen result

| Metric | Exact v2 control | v2 + deterministic RHT |
|---|---:|---:|
| Logical payload sum | 17,916,908 bits | 18,006,314 bits |
| Mean logical payload | 559,903.3750 | 562,697.3125 |
| Reservoir headroom | 113,940 bits | 24,534 bits |
| Physical reservoir size | 18,033,152 bits | 18,033,152 bits |
| Physical rate | 2.14971923828125 bpw | 2.14971923828125 bpw |
| Energy-weighted relative MSE | 0.06319873774126093 | 0.05289448474927123 |
| Excess over Gaussian limit | 24.4908% | 4.19320% |
| Blocks below the 5% ceiling | 27/32 | 32/32 |
| Independent decodes | 32/32 | 32/32 |
| Joint rate/MSE gate | Fail | Pass |

The Gaussian reference at 2.15 bpw is
`2^(-2*2.15) = 0.050765774772264724`; the declared ceiling is
`1.05 * reference = 0.053304063510877964`. RHT reduces MSE by
`16.3045234%` relative to the same-source exact control.

The distribution audit should reproduce:

- raw kurtosis median `3.3754747491`, maximum `114.2917733135`;
- RHT kurtosis median `3.0014572220`, maximum `3.0194622728`;
- correlation of exact-v2 MSE with raw kurtosis `0.9952096176`;
- correlation of RHT MSE with transformed kurtosis `-0.1212068101`.

The generated summaries contain absolute paths and wall-clock timings, so
their whole-file hashes are not expected to match the checked-in summaries.
On the pinned environment, the two generated reservoirs are expected to be
bit-identical to the checked-in reservoirs and to have the hashes shown
above.

## Reproduce the Gaussian confirmation

The synthetic confirmation workspace preserves the original filenames and
hash bindings under `frozen/gaussian_confirmation/`. Its harness uses a
one-time opened lock. The safe wrapper copies that complete tree to a new
destination, injects the locally generated hash-checked decoder map, and then
invokes the historical-basename harness, so the archival copy stays pristine:

```bash
"$PYTHON_BIN" tools/verify_gaussian_confirmation.py \
  --workspace "$PWD/repro/gaussian_workspace" \
  --polar-repo "$PWD/third_party/PolarLatticeQuantization" \
  --run-dir "$PWD/repro/gaussian_run" \
  --summary "$PWD/repro/gaussian_summary.json" \
  --python "$PYTHON_BIN" \
  --workers 4
```

Do not delete the opened lock to rerun or turn a partial run into a new
confirmation. Make another clean copy and use a different run directory. A
successful run emits `repro/gaussian_run/confirmation.plrsv2` with SHA-256
`ad0c35e72b5900ffa6ed353df1bf1b163d912b8bfb692fc8e5b318ea6f9eb3f5`.
The published statistical results and confidence calculation are documented
in `docs/GAUSSIAN_CONFIRMATION.md`.

## Frozen hashes

These hashes identify the normative Qwen implementation and compact evidence.
Renaming a byte-identical file does not change its hash. Support tools used
only to download sources are listed separately because they do not define the
codec bitstream.

### Codec and evaluation implementation

| Path | SHA-256 |
|---|---|
| `src/polaris_sc_v2_encoder.py` | `95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8` |
| `src/polaris_sc_v2_rht_encoder.py` | `062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0` |
| `src/reservoir_pack_v2.py` | `c5fda34242153365dac07b5990bcd1fa19f0ac98d2512d47c3c8e1ec2a81dde8` |
| `src/reservoir_unpack_v2.py` | `cf7113c3fbc6340f0870dadcf7608739aa651f5706befa163b5d13516dac7e07` |
| `src/independent_decoder_v1.py` | `0652521fae5c77567e67cc9434adfb84b1a3cab53e5a79f250a3093a467be072` |
| `src/agent_polaris_independent_decoder_v1.py` | `0652521fae5c77567e67cc9434adfb84b1a3cab53e5a79f250a3093a467be072` |
| `src/qwen_reservoir_decode.py` | `2e1e484bf8ba98d493cfda55d4b23e275267e097e08907f5a9c606ae7350c797` |
| `codec_data/polaris_sc_v1_decoder_map.npz` (locally generated; not tracked) | `a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef` |
| `tools/run_qwen_panel.py` | `4229ffd0d1fd43211ba2ad3022f1c684e452e4df158992493bd0f879663b3d59` |
| `tools/audit_qwen_paired.py` | `a29e4bd38dc5da88552d964af765e635cdd3571b4b3b4b160e361f36a2290943` |
| `tools/audit_qwen_distribution.py` | `643303fa4080590c409a55f6b1dc690d1fb68a36ce8c72b332345da77b6020ea` |

### Published Qwen evidence

| Path | SHA-256 |
|---|---|
| `results/qwen/manifest.json` | `3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55` |
| `results/qwen/exact_summary.json` | `8b12e5bdbb8a58c4890637e7250697bcabdb83564037d9eb1e65a563bd7736a9` |
| `results/qwen/rht_summary.json` | `db4a70ab1d7dcf246d0232d580e329e3675b152f0f9c6232f3583434e43f3f5a` |
| `results/qwen/paired_audit.json` | `666fa77eb17fc9fda2c27f3081913b7a383d103acaf9b92749fcbbe921bcfedc` |
| `results/qwen/distribution_audit.json` | `f4724f0bd1118074b28ffed50d635a75640c5b926bf70777bfe7e1dc3e437fa0` |
| `results/qwen/independent_audit.json` | `52fc0af9cb95ccf6a0436f1a5710e3792f2b88e587070e27faec16149928029e` |
| `results/qwen/independent_audit.md` | `e2ee14f50dac69901c7d50e2cf4735081db19a0bd46295250d33d01145c374d9` |
| `results/qwen/independent_audit_erratum.json` | `53660b5f3f3faee4c9c0d90fef70ac8c1d978800509029b72cd492b88a925738` |
| `results/qwen/independent_audit_erratum.md` | `3385f40faf01da94b24f6dcc488a1ecb8b58c038f5590ec8f5d1b4b1ae587da1` |
| `results/qwen/exact_pack.json` | `de22f1739bfc08164aa7faa60aed98af44cc6a15b7022821a4e73045c1f568c4` |
| `results/qwen/exact_unpack.json` | `0be31fdac57ae982f8fe06c98474d0fe566dd9ede441f599af027ddd09d104bb` |
| `results/qwen/rht_pack.json` | `6451d8c74bece6cab79dffa6f705a317cb5e77f680235455cce8837019334165` |
| `results/qwen/rht_unpack.json` | `1d8c976f326394032e2339aa51bcb921bb535ca816f08ddfb949e830fb047b09` |
| `results/qwen/exact_panel.plrsv2` | `9388790c3cdbab5b9b33b676ced196090d81ba0422eb6fecfdd014bd2d054cf5` |
| `results/qwen/rht_panel.plrsv2` | `55d347c02ef1382ce209050d539f4e336dd7477125e4319e8b78d3067a436aac` |

Support-tool hashes at publication are
`733671ae48c977216b5ee6b21e5abfb406f1cd2c5de7887955917ca07e180fe0`
for `tools/fetch_qwen_block.py` and
`af0670112d738ad6b7621224707c2272f3eb9474cff07652a81a41521d789aa1`
for `tools/materialize_qwen_panel.py`.

## Fail-closed behavior and troubleshooting

- **Implementation hash mismatch:** use the byte-frozen files from this
  revision. The runner intentionally rejects even a harmless edit to a
  normative script.
- **Decoder map missing:** review the upstream terms, clone the pinned commit,
  and run `tools/build_decoder_map.py`. The asset is deliberately untracked,
  and every full decoder rejects a wrong hash or geometry.
- **Manifest or source hash mismatch:** confirm the checkpoint revision and
  C-order block index. Never redraw or substitute a convenient block.
- **HTTP range not honored:** the fetcher aborts rather than downloading an
  entire shard. Retry later or use a mirror that preserves the same immutable
  bytes and verify the manifest SHA-256.
- **Output directory already exists:** select a new directory. The unpacker
  and runners refuse accidental overwrite, resume, or result relabelling.
- **Exact arm exits 2:** this is expected only after it writes a complete
  summary with `passes_joint_rate_and_mse_gate: false`. Any earlier exception,
  missing summary, or other exit status is an execution failure.
- **RHT arm exits nonzero:** treat it as failure. Do not change workers, seeds,
  block membership, or codec parameters to rescue the run and then call it the
  frozen result.
- **Different reservoir hash:** first compare Python, NumPy, SciPy, CuPy,
  CUDA, GPU, upstream commit, all implementation hashes, and all 32 source
  hashes. Do not compare summary-file hashes because paths and timings vary.
- **Import error for the independent decoder:** retain the historical
  compatibility filename in `src/`; the two decoder-core files must be
  byte-identical.

## Claim limitations

This reproduction establishes a strict PTQ weight-codec result on the frozen
32-block Qwen coverage panel and a separately confirmed matched-Gaussian
source-code result. It does **not** establish:

- a full 30.5B-parameter checkpoint encode or an emitted full-checkpoint
  reservoir;
- a probability-sample estimate or confidence interval for all Qwen weights;
- perplexity, downstream accuracy, generation quality, activation-weighted
  error, or router-frequency-weighted quality;
- production encode/decode throughput or an optimized inference kernel;
- transfer to another Qwen revision, model family, source dtype, or block
  size;
- a universal improvement over all contemporary 2-bit quantizers.

The tested 8,388,608 values are about `0.0275%` of the checkpoint's rank-2
weights. The manifest deliberately spans all rank-2 matrix families, but it
is not fully blinded: embedding block 0 had been used for a pipeline/RHT
round-trip smoke. Expert 31, which was used in development, is excluded from
the expert portion of the panel. No quantizer parameter, panel member, seed,
or retry was changed after the panel outcomes were opened.

The next confirmatory step is the preregistered probability sample described
in the research notes or a complete 116,470-block rank-2 census, followed by
whole-model reconstruction and perplexity/downstream evaluation.
