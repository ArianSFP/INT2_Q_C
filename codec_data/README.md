# Locally generated decoder asset

The codec expects this generated file in this directory:

```text
polaris_sc_v1_decoder_map.npz
```

It is intentionally not redistributed because it derives from reliability
tables in the pinned `graceBaoXP/PolarLatticeQuantization` repository, which
does not expose an explicit license. Generate it from a separately obtained
checkout after reviewing the upstream terms:

```bash
python tools/build_decoder_map.py \
  --polar-repo third_party/PolarLatticeQuantization \
  --output codec_data/polaris_sc_v1_decoder_map.npz
```

The builder refuses a different upstream commit, encoder implementation, or
output hash. The required result is 200,184 bytes with SHA-256:

```text
a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef
```
