# Source-free smoke supersession

The first source freeze had manifest SHA-256
`cf5740c0bf1a378db102220c31df025ca6c7ae7802dbaecdec55fcd9d905aee1`
and source root
`063182d1f4025680b17d706d924d0af8ac3ed45202cfa4018b958bea47af4783`.

Its 25 standard-library tests and verifier passed, but its first external
source-free CuPy smoke stopped before emitting a record:

```text
NotImplementedError: axis option is not supported yet
```

The failure came from calling CuPy 14.2 `packbits(..., axis=1)`. No Qwen/model
payload or live v6 result was accessed. That failed freeze has no launch or
result authority.

The repaired freeze uses
`packbits(signs.reshape(-1), bitorder="little").reshape(blocks,47)`. Since
each row is exactly `376 = 47*8` bits, there is no partial-byte boundary and
the flattened operation is exactly the frozen per-record bit grammar. No
codebook, scale law, rate, selector, gate, or scientific threshold changed.

The package manifest following this note supersedes the failed first freeze.
