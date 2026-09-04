# COCHAIN-Q plaquette/cube source-only oracle

## The exact scientific distinction

For a binary label field on a complete `2x2` plaquette, the mixed derivative is

```text
s = q00 xor q01 xor q10 xor q11.
```
For a `2x2x2` cube it is the XOR of all eight vertices.  Keeping the first
`V-1` labels and `s` is an invertible change of coordinates.  It still costs
exactly `V` bits.  **Fixed cochain differencing alone cannot save one bit or
change MSE.**

COCHAIN-Q becomes a real quantizer only when it uses a smaller codebook.  This
v0 package fixes the syndrome to the public value zero, omits it, and searches
all `2^(V-1)` legal label fields for minimum exact distortion.  It therefore
uses:

- `3/4 = 0.75` bit per bitplane site for a plaquette;
- `7/8 = 0.875` bit per bitplane site for a cube.

The reduction is paid for by any distortion needed to move nearest labels into
the public fiber.  The reported equivalent gain is the single physical score

```text
(R_baseline - R_fiber) + 0.5*log2(D_baseline/D_fiber).
```

No separately measured rate and MSE gains are added.

## What is exact

- Every 2D and 3D cell pattern is enumerated.
- A literal global brute-force implementation independently checks tiny cells.
- The boundary-plus-syndrome map is exhaustively proven bijective over all 16
  plaquette and all 256 cube patterns.
- The public-fiber packet literally stores `V-1` bits and derives the final bit
  at decode.  Nonzero padding is rejected.
- A source-adaptive choice between odd and even fibers is diagnostic only and
  is charged one selector bit.  Only the fixed even public map is eligible for
  promotion.

The public packet is entirely expert-local.  It has no common expert stream,
no neighbour fetch, and logical routed-read amplification exactly `1.0x`.
Page rounding is reported separately and converges to `1.0x` for production
expert sizes.

## Fixtures

The low-degree ensemble contains every Boolean truth table whose highest-degree
monomial is absent.  Equivalently, its complete mixed difference is zero.  It
has zero pairwise mutual information but saves exactly one bit per cell:

- `0.25 bpw` for plaquettes;
- `0.125 bpw` for cubes.

This proves that COCHAIN-Q detects a pure higher-order constraint invisible to
pair tests.  A balanced IID ensemble has an exactly uniform syndrome.  The
invertible representation saves zero, and a high-confidence public-fiber
quantizer is correctly hard-killed because enforcing the constraint is costly.

## Qwen pilot, if separately authorized and audited

A local RTX 3060/CuPy pilot would consume exact legal distortion costs from the
real deployed STRATA representation, never frozen nearest labels alone.  For
each of the six label planes, each role, traversal, and public cell tiling it
would compute in chunks:

1. the nearest-label SSE;
2. exact minimum SSE in the even plaquette and cube fibers;
3. the literal packed rate including padding and any fixed headers;
4. the one composite equivalent gain above;
5. moment-matched Gaussian controls repeating the complete selection;
6. whole-expert and whole-layer confidence intervals;
7. an independent CPU re-score of the selected labels and packet decode.

Candidate tilings are frozen on auxiliary experts/layers.  Selecting a tiling,
syndrome, role subset, plane subset, or traversal on evaluation payloads is not
free.  The Qwen run must bind the exact local GPU UUID/runtime and source hashes
in a different one-use capability package.  This source package grants no such
authority.

Strict gates after complete physical charging and Qwen-minus-Gaussian control:

- `<0.045 bpw`: hard kill the tested public plaquette/cube family;
- `0.045-0.10 bpw`: real but insufficient alone;
- `>=0.10 bpw`: eligible for one nested finite composite;
- `>=0.15288996696 bpw`: standalone rate-equivalent route to `F<=0.8`;
- `>=0.18 bpw`: margin sufficient for six-plane engineering.

The cube ceiling is only `0.125 bpw` per affected bitplane site, so a plain
single cube constraint cannot be a standalone full-weight breakthrough.  A
plaquette can reach `0.25 bpw`, but only if label movement costs little.

## Boundaries and limitations

This is a binary bitplane oracle, not a legal six-plane STRATA packet.  It does
not test overlapping plaquettes, multiscale RENORM-Q, learned factors, posterior
centroids, Gate/Up/Down cube geometry, or cross-plane carry constraints.  The
public zero syndrome is deliberately minimal and may be a poor match to real
weights.  A positive source-only fixture is mechanism validation, not Qwen
evidence and not a deployable codec.

Run from the repository root:

```powershell
C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B `
  research\cochain_q_plaquette_v0_20260904\verify_source.py `
  --package research\cochain_q_plaquette_v0_20260904 --self-test
```
