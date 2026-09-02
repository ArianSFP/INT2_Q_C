# Independent source audit: ε-TCQ compact polar memory v2

This package independently authenticates and recomputes the source-only v2
capacity result. It imports none of the candidate's memory/work functions.

Pinned source:

- manifest SHA-256: `cef51a7a62619927503749ebf3a390241aa9297842480023b3ffcc5abd4cf277`
- source root: `92c7969cddbebf255c19f1aa10869d704c68727a8562f8f47bd27dd4c3593ff4`
- authenticated STRATA decoder: `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e`
- frozen RunPod CuPy receipt: `083979b2531066e0a81f4bec3a9afa5dd027d4cd934b6fb9ce240491fa099c14`

The audited split verdict is deliberately narrow:

- `GO_MEMORY_CAPACITY`
- `HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION`
- `HOLD_PAYLOAD`

The RTX 5090 receipt proves all 34 frozen allocations coexist for beams
4/8/16/32. It does not prove an exact Q0.16-boundary GPU implementation,
device-resident COW, persistent six-level SC execution, or useful throughput.

No Qwen, current-codec or Gaussian payload is accepted or accessed.
