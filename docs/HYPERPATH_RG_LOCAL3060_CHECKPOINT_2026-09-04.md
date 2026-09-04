# HYPERPATH-RG local RTX 3060 checkpoint

Date: 2026-09-04

Execution boundary: every payload experiment in this checkpoint ran on the
local NVIDIA GeForce RTX 3060
`GPU-458a424a-76e3-65e5-0470-803e0ed131ca`.  No RunPod or network execution
was used.

## Goal and unchanged best checkpoint

The target remains one independently decodable, universal SwiGLU-MoE packet
at `2.15 <= R <= 2.5 bpw`, pooled original-weight
`F = relative_MSE * 2^(2R) <= 0.8`, and maximum routed cold-read amplification
strictly below `2x`.

The best deployable finite checkpoint remains:

```text
R = 2.5 bpw
D = 0.030902167403153148
F = 0.9888693569009007
```

It is 1.1131% below the Gaussian reference.  It still needs 8.98697% lower MSE
for the `F=0.9` checkpoint and 19.09953% lower MSE for `F=0.8`.

The ideal structural composite remains `F=0.93639762`, not a finite codec.  It
needs another 3.88698% MSE reduction for `F=0.9` and 14.5662% for `F=0.8`.

## PAIRPATH audit and stronger bound

Independent audit blocked PAIRPATH r2 for three reasons:

- finite Up/Down searches used role-local rather than one global multiplier;
- a legal counterexample beat the claimed joint solution by
  `0.07489358865` objective units;
- the decoder accepted an invalid unreplayed tree descriptor.

Audit manifest:
`5d1e577fb10e0803f22c218ac55c72b907b8f2cf0713c49993dbcfe2880fb4ef`.

PAIRPATH r3 repairs the finite multiplier and tree replay and inserts a
candidate-dominance certificate.  It correctly retains
`hard_kill_authority=false` because its alternating search is not globally
optimal.  Source manifest:
`4f0ab10ff813860dc4e4512079f8eb17fc0987a1e7ca438cb5d90a858a1453cd`.

A stronger globally optimized stochastic aperture made a Qwen PAIRPATH run
unnecessary.  The exact same-role cross-expert topology
`(Up_e,Up_f)+(Down_e,Down_f)` gained at most `0.00316455 bpw` and averaged
`0.00283499 bpw`, even with free per-block laws.  This is far below the
`0.045 bpw` continuation threshold, so memoryless cross-expert PAIRPATH is
stopped.

## TETRAPATH-4 result

The source-only XOR fixture is valid: pure four-way parity saves `0.25 bpw`
with zero pairwise mutual information, and its common/private fiber has ideal
logical read amplification `4/3`.  The original heuristic optimizer was not a
valid hard-kill oracle; its independent audit found a missed exact solution,
non-containing smoothing, and incomplete factor attribution.

Two Qwen measurements replaced it:

1. Nearest-label census over all eight fixed layer-15 expert pairs:
   maximum raw four-way gain `0.00003870032 bpw`; maximum permutation-corrected
   gain `0.00001369844 bpw`.
2. CuPy Blahut--Arimoto relaxation over a 1/64 block aperture of all eight
   pairs.  Each block received a free stochastic 256-state law and free time
   sharing.  Full-vs-independent gain averaged `0.06235430 bpw` raw and
   `0.02861814 bpw` after the destroyed control.  However, the largest raw
   full-vs-best-2+2 gain was only `0.03087325 bpw`; the best corrected value was
   `-0.00028168 bpw` and the mean was `-0.00187063 bpw`.

Thus the apparent four-way survivor was entirely pair-factorizable.
Irreducible coordinate-memoryless TETRAPATH and its fiber packet are stopped.

Authoritative results:

- fixed-label result:
  `2dab5c4175149d92f46409724bc2d204e05452e5072efcbb23cffad9a1f19418`;
- BA all-eight result:
  `3b68c4ee7115bfb8d5f6b6e8027a2bb27c5c0f6d358647b4c4394182e7158353`.

The best 2+2 factor was overwhelmingly expert-local Up/Down rather than
cross-expert.  Its control-corrected gain ranged `0.0102643--0.0521481 bpw`
and averaged `0.0316639 bpw`.  This is a useful `~1x`-read clue, but far below
the `0.22933495 Up/Down bpw` standalone requirement.

## True six-plane STRATA-RM6 result

The first bounded local-3060 Qwen pilot used one authenticated 4,096-value
layer-15 expert-0 Up block, the real 64-way distortion table, six complete
planes, exact RM(5,12) generator moves, and the literal arithmetic-length
check.

The best zero-coset row reduced its own RM-SC initializer MSE by `5.23756%`.
The identical moment-matched Gaussian path improved by `7.07671%`, leaving
`-1.83915` Qwen-specific percentage points.  Its selected stream was 1,408
bytes (`2.75 bpw`); all nine Qwen checkpoints exceeded the strict 2.5-bpw cap
and the packet codec rejected them.  The tested local-bank coordinate-descent
route is stopped.

Result SHA-256:
`fa99e07439e6381443f6681001caed96d61838649a9b974d7bc07468600f7e00`.
Package manifest:
`51fc887824754852a5d7cf2a5b7b6e6814d098db1d5f91568faf6c102efdc1f6`.

This bounded result is not transferable to deployed global STRATA and is not
a converse for every RM6 construction.

## RENORM-Q and COCHAIN-Q source gates

RENORM-Q v0 implements a small public collective-map bank and exact tree
min-sum that matches exhaustive tiny enumeration.  Its XOR/IID controls pass,
but independent audit blocks Qwen because the v0 API accepts non-Kraft NLLs,
permits caller maps to self-declare zero descriptor cost, contains unreachable
declared states, and applies its LCB gate incorrectly.

COCHAIN-Q v0 correctly distinguishes invertible zero-saving differencing from
a real public syndrome-constrained codebook.  Its exact ceilings are `0.25 bpw`
per affected plaquette bitplane site and `0.125 bpw` per cube site, with ideal
expert-local `1x` reads.  Independent audit blocks Qwen because of manifest
closure/pinning, dtype coercion, and full-expert normalization defects.  A
valid future result must emit one legal six-plane reconstruction and pool its
source SSE; per-plane gains cannot be added.

Both branches remain mechanism research, not Qwen evidence or codec gains.

## Decisions

| Branch | Decision | Reason |
|---|---|---|
| Fixed-label same-layer CBIB | Stop | physically charged gain negative |
| Memoryless cross-expert PAIRPATH | Stop | stochastic upper aperture max `0.003165 bpw` |
| Irreducible memoryless TETRAPATH | Stop | raw full-vs-2+2 max `0.030873 bpw`; controls negative |
| Local RM(5,12)^6 bank 0 | Stop | 2.75+ bpw and worse than Gaussian control |
| Intra-expert Up/Down flexible pairing | Hold as clue | `0.031664 bpw` mean, insufficient alone, ~1x reads |
| RENORM-Q | Repair source contract before Qwen | multiscale hypothesis remains open |
| COCHAIN-Q | Repair source contract before Qwen | spatial flexible-label hypothesis remains open |

No gains above are additive.  The `-10%` and `-20%` finite checkpoints have
not been reached.
