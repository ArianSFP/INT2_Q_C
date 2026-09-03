# TACTIC Ramanujan-384 scalable v2 source assessment

## Authenticated scope

- Producer manifest SHA-256: `1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209`
- Producer source root SHA-256: `bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495`
- External verifier SHA-256: `74f5a56f1371f67ffa4e83ea34b761c2de61ea0900e3374cc25092f2d333e92c`

All fourteen declared producer members were independently read, length-checked,
hashed, and recomposed into the pinned canonical root. The package is an exact
flat closure at review time. No payload-like input was accessed.

## Backend-independent Gaussian controls

The v1 Irwin--Hall construction is replaced correctly in source. SplitMix64
maps absolute counters to two independent 53-bit midpoint uniforms; Box--Muller
is evaluated on the host; results are rounded to little-endian IEEE binary32
and widened to little-endian binary64. `moment_matched_gaussian` performs the
moment match on host arrays and copies those serialized values to the selected
backend. Neither backend RNG nor backend transcendental functions participate.

The CuPy runner explicitly byte-compares both a canonical Gaussian round trip
and separately produced CPU/CuPy moment-matched arrays. Thus NumPy and CuPy in
one pinned host environment receive identical control bytes. Cross-platform
libm bit identity is not proved: binary32 canonicalization makes last-bit drift
unlikely to survive, but it is not a correctly-rounded transcendental
specification. This does not weaken the same-host CPU/CuPy contract.

## Batched search and synchronization

For each role, the source selects one ordered fourteen-atom support and builds
all rank-1 through rank-14 prefix systems in a single tensor. It contains one
`xp.linalg.solve` call and one candidate reconstruction `einsum`. Winner
metadata crosses to the host in one bulk array and aggregate SSE/histogram data
in a second bulk array. There is no `.item()` call or per-candidate Python
solve/matmul in the hot core. Literal packet encoding, decoding, and replay
occur after winner selection and are checked against the chosen canonical
coefficient state.

This is a batched search of the fifteen rank prefixes, not an exhaustive search
over all support combinations. The reported `per_candidate_host_scalar_syncs =
0` is supported by the source structure; only an actual CuPy profiler can rule
out backend-internal synchronization.

## Target-rate fixture and controls

The `[128,2048]` three-role fixture gives 786,432 weights, 235,776 coarse bytes,
9,216 fine bytes, a 512-byte header, and 256 bytes of final page padding. The
245,760-byte object is exactly 2.5 bpw. Its public block-reset Ramanujan atoms
are designed to survive the absolute-D gate. When that gate passes, the adapter
mandates one phase-permutation control and all eight Gaussian controls through
the same batched encoder.

The frozen CPU receipt reports literal replay, all nine controls, relative MSE
`9.5367431640625e-7`, `F = 3.0517578125e-5`, and 10 bpw source gain. This is a
deliberately constructed periodic source with a synthetic zero coarse decoder.
It demonstrates mechanism and control reachability only. It is not Qwen
evidence, a 10-bpw compression result, or evidence that a model residual has
Ramanujan structure. The requested CuPy target-rate execution remains pending.

## Shape, count, packet, and tail bounds

Dimensions are positive uint32 values; per-role block count must fit uint32;
coarse and fine lengths must fit uint64; the exact 307/128-bpw coarse length
must be byte-integral; and page rounding is guarded before addition. The coarse
integrality condition forces role values to a multiple of 1,024, so every tail
has at least 1,024 valid values. Tail masks enter atom norms, Gram matrices,
candidate SSE, replay SSE, and input SSE. Header block counts and payload
lengths are rederived from dimensions during decode.

The caps prevent serialized-field overflow. They are not practical allocation
limits: extreme but representable shapes can still request infeasible host or
device memory. A production launcher needs an explicit resource budget.

## External closure

The separately pinned bootstrap verifier imports no package code. It checks the
manifest hash and schema, every member length/hash/type, single-link status for
members, canonical row order and root, exact directory entries, and pinned v1
dependencies. This authenticates the frozen flat closure at the instant of its
check.

The verifier does path/stat/read operations rather than descriptor-based
snapshot reads and does not launch the CuPy runner. The runner itself checks
only the manifest-file hash before importing members; it does not recompute the
member hashes/root. Consequently a file swap after bootstrap validation and
before import is not detected. The manifest file is also not required to be
single-link. For benign frozen-source use this is a narrow operational limit;
production execution should copy verified bytes into an immutable snapshot or
have the external verifier exec the runner from already-open authenticated
descriptors.

## Independent coarse-decoder capability

The capability mechanism is materially stronger than v1. It opens and hashes
the capability, decoder source, decoder manifest, auditor manifest, and PASS
receipt; checks the runtime object's class-source path and capability ID; gives
the decoder only coarse bytes and public geometry at the call site; and rejects
decoded arrays unless each role matches an independently recorded FP32 hash.

It does not itself prove an independent or source-blind runtime:

1. `authenticate` accepts a caller-supplied live Python object. An instance
   method can be monkey-patched or instance state can influence decoding while
   `inspect.getsourcefile(type(decoder))` still names the pinned class file.
2. A module can be loaded before its file is replaced with the pinned bytes.
   The file hash and source path then authenticate bytes different from the
   already-loaded class.
3. The decoder executes in the same unrestricted Python process. Passing only
   coarse bytes as arguments does not prevent file, global, or network access.
   Source blindness is a receipt assertion, not an enforced capability.
4. The decoder and auditor "manifests" are parsed, but their schemas, roots,
   and exact closures are not validated; only the decoder's named source row is
   checked. Receipt fields are not exact-schema checked.

The independent output hashes strongly constrain ordinary accidental errors,
but do not close source leakage if the decoder can read the same reconstruction
artifact or source. A real production capability therefore remains mandatory
and should load authenticated source into a fresh restricted process, forbid
instance monkey-patching/state not represented in the capability, enforce no
source/network handles, and bind an independently generated output-hash
receipt.

## Disposition

The scalable search, Gaussian-control input construction, rate/control path,
shape-field caps, and at-review-time external closure pass static review. The
producer's source-free CPU fixture is credible as a mechanism test. There is no
CuPy or Qwen result in this review. Payload promotion remains blocked on the
pending CuPy execution and a real hardened coarse decoder; atomic closure-to-
execution binding is also required for hostile reproducibility.

```text
PASS_V2_STATIC_SCALABILITY_AND_SOURCE_FREE_CONTROL_MECHANISM__HOLD_CUPY_QWEN_AND_PRODUCTION_COARSE_DECODER_UNTIL_ATOMIC_CLOSURE_AND_HARDENED_EXECUTION
```
