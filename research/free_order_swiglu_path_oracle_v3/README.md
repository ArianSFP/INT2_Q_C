# FOSP-ARX v3-DIRECT-SEALED: source-only blocker repair

## Verdict

This is a distinct, deployment-blocked successor to the immutable v2 package.
It preserves v2's full cross-role `3 x 3` predecessor/path family and repairs
all three findings in the sealed independent v2 audit. No source binding was
resolved; no Qwen/model, pinned, validation, runtime-evidence, authorization,
or production payload was opened; no NumPy/CuPy/SciPy/Torch module was
imported; and no CUDA, GPU, network, or remote action was performed.

V3 is source-only. It contains neither a calibration entrypoint nor an
authorization builder. A new independent v3 source audit and an independently
sealed interpreter/runtime closure are prerequisites to any later work.

## Scientific repair: corrected relaxed reuse is not containing

For a joint SwiGLU neuron `Y` and a distinct predecessor `X`, both shaped
`3 x 2048`, the oracle retains the exact v2 predictor:

```text
A*      = (Y X^T) (X X^T)^-1
capture = tr((Y X^T) (X X^T)^-1 (X Y^T)).
```

All `768 * 767` ordered nonself pairs are scored in float64. The achievable
path remains a maximum-weight nonself cycle cover, a deterministic cut at the
weakest incoming edge of every cycle, deterministic segment ordering, and
explicit bridges. All nine coefficients on all 767 selected edges are rounded
through IEEE binary16 and the residual is replayed directly. The permutation
is still serialized as a 783-byte factoradic rank.

V3 distinguishes two quantities that v2 conflated:

- Gross Qwen target-wise relaxed reuse contains every legal exact path, so it
  is a deterministic necessary bound and may hard-kill when it misses.
- `s_qwen_relaxed - mean(s_control_relaxed) + 3 SE` does not contain the
  corresponding corrected legal statistic. V3 reports it only as a diagnostic;
  it can neither kill nor promote.

Every Qwen/control panel computes its achievable legal exact and legal FP16
statistics. After controls exist, corrected legal FP16 is computed before any
control-corrected decision. A legal FP16 miss remains an optimization-gap
ambiguity because the cycle-cover construction is not claimed optimal.

### Exact n=8 regression

The hostile audit's realizable construction is frozen as a test. Three roles
occupy orthogonal zero-mean subspaces. Q uses AR(1) correlation `r=7/8`; each
control uses the star geometry with hub/leaf `r` and leaf/leaf
`rho=r^2=49/64`. Both coefficients round-trip exactly through binary16.

```text
corrected relaxed s = 0
Q legal FP16 s      = 0.7995602818589078
control legal FP16  = 0.5885652320580218
corrected legal     = 0.21099504980088601
required gross s    = 0.1858070514584381
```

The legal survivor therefore remains visible; no corrected-relaxed hard kill
exists anywhere in v3.

## Rate and read science preserved

One expert has `3 * 768 * 2048 = 4,718,592` weights.

| Field | Bits |
|---|---:|
| 64-byte header | 512 |
| 783-byte factoradic permutation | 6,264 |
| `767 * 9` FP16 coefficients | 110,448 |
| Total side information | 117,224 |

The total side charge is `0.024843004014756944 bpw`, raising the gross target
from `0.16096404744368115` to `0.1858070514584381 bpw`. Logical expert-frame
read amplification remains `1.0x`; the worst frozen cold-page accounting is
`1.0054349308378698x`, below `2x`.

## Pre-import package and runtime closure

The only security-bearing entrypoint is `bootstrap_v3.py`. Invoke it with an
independently trusted, externally hash-pinned Python executable in isolated,
no-site mode:

```text
python -I -S bootstrap_v3.py \
  --package-manifest-sha256 <externally pinned digest> \
  --verify-package
```

Before importing any filesystem module, the bootstrap:

1. imports only `sys` and the platform's built-in `nt`/`posix` primitive;
2. verifies `-I`, `-S`, and safe-path mode, then empties `sys.path`;
3. enumerates the flat package and rejects every directory, symlink/reparse
   point, socket, FIFO, device, or other nonregular object;
4. hashes the externally pinned manifest with an internal dependency-free
   SHA-256 implementation;
5. opens, identity-checks, sizes, hashes, and retains every sealed member;
6. executes an entrypoint only from those already captured bytes.

Thus a top-level `json/__init__.py` directory is rejected before its code can
execute. The regression suite copies the exact package, installs precisely
that injection, and proves the sentinel remains absent.

Runtime execution mode additionally requires an externally pinned Python
executable digest and an externally pinned runtime manifest. That manifest
declares every directory, every regular file size/hash, and the only allowed
source roots. The bootstrap enforces exact recursive closure, forbids runtime
links/special objects, retains all authenticated runtime source bytes, removes
the normal filesystem finder, and imports Python modules only from that
in-memory snapshot (plus interpreter built-ins/frozen modules). The package
directory is never importable. Native-extension/GPU execution is deliberately
outside this source-only v3 boundary.

This mechanism cannot make a compromised interpreter attest itself. The
interpreter and the two root digests are explicit external trust inputs. That
trust boundary is now stated rather than hidden behind a self-generated JSON
seal.

## Audit-verifier binding repair

A future audit receipt is acceptable only when an external trust root pins the
exact audit-manifest digest. The held audit directory must have exact object
closure; every listed member, including `verify_audit.py`, must be opened and
hashed. Merely seeing the literal row name is forbidden. The verifier must run
through the same independently sealed interpreter/runtime closure. Duplicate
JSON keys and nonfinite JSON values are forbidden throughout.

V3 does not include an authorization builder, so this source package cannot
mint a synthetic trusted audit digest or authorize itself.

## Source-only verification

Using the manifest digest printed by a separately trusted hash tool:

```text
python -B -I test_source_only.py
python -B -I verify_package.py --manifest-sha256 <digest>
python -I -S bootstrap_v3.py --package-manifest-sha256 <digest> --verify-package
```

All commands are standard-library/source-only. They must not be augmented with
model paths, runtime/calibration evidence, authorization files, GPU visibility,
or network access.
