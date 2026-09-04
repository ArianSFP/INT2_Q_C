# STRATA-RM6 true six-plane Qwen pilot — local RTX 3060

## Result

`HARD_KILL_BANK0_QWEN_PHYSICAL_OVERFLOW_AT_ALL_CHECKPOINTS`

This is the first bounded Qwen payload test of the existing audited
`strata_rm6_label_flexible_gate_v0` mechanism.  It uses the real 64-way
distortion table, six complete level-major STRATA planes, exact RM(5,12)
generator moves, the literal Q0.16 arithmetic model and the candidate's
canonical packet decoder.  It ran only on the pinned local RTX 3060.

The strongest Qwen row was the zero-frozen affine coset:

| quantity | value |
|---|---:|
| Qwen block | Qwen3-30B-A3B, layer 15, expert 0 Up, block 191 |
| values | 4,096 |
| unconstrained 64-way relative MSE | 0.0052110063840 |
| legal RM-SC initialization relative MSE | 3.7063403561 |
| selected legal local-search relative MSE | 3.5122185748 |
| Qwen MSE reduction versus identical RM-SC initializer | 5.2375594965% |
| matched-Gaussian reduction through identical path | 7.0767066333% |
| Qwen minus control | -1.8391471368 percentage points |
| Qwen F-equivalent improvement inside this pilot | 0.0388063704 bpw |
| selected checkpoint | 29 legal generator flips |
| selected logical arithmetic bits | 9,973 |
| derived aligned packet | 1,408 bytes = 2.75 bpw |
| literal packet emitted | no; the immutable codec correctly rejected it |

The current-random affine coset was worse: its RM-SC initialization had
relative MSE 13.698418474 and a derived rate of 3.25 bpw.  Distortion-only
legal flips increased arithmetic rate, so the lowest F checkpoint remained
the initialization.  The matched-Gaussian current-random row behaved the same
way.

The zero-coset Qwen gain is below the locked 10% panel-promotion gate, is
smaller than the identical Gaussian-control gain, and has no physical packet
at or below 2.5 bpw.  Expanding this exact bank-0 coordinate-descent pilot to
more Qwen blocks is therefore rejected.

## What was compared

The comparator is a source-conditioned, single-path RM-SC MAP initialization
under the same local RM bank, coset and source model.  The candidate applies
up to 128 CuPy FP64 legal RM generator-row flips and selects the lowest-F
predeclared checkpoint after recomputing exact CPU FP64 source SSE and literal
aligned arithmetic rate.

This is **not** a comparison against the deployed global STRATA result.  The
local RM(5,12)^6 topology is a new 4,096-coordinate code, while deployed
STRATA uses 2^20/2^21 polar blocks and a different frozen-set schedule.  The
very high local legal-codeword MSE must not be transferred to the deployed
codec.

## Authentication

- immutable STRATA-RM6 source manifest:
  `c8d56e045159e3af613f02c4d5d97c70e8f8b4383b3fbf282d384b08f74b7300`
- immutable source root:
  `d17718615dedebca08ead66c0555e9d649768a353f3a55d169a9bf400f11bd32`
- static independent audit-source manifest:
  `3955d339322ef175f16fefd2f74d327faa800cb0bdd34bb6b1f0952e7905984d`
- Qwen panel lock:
  `1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5`
- Qwen BF16 payload:
  `15e0782da2120ab5369ec88bd3d4ae3a677ed0e63b38b48f2f622a72d1f574b6`
- result file:
  `fa99e07439e6381443f6681001caed96d61838649a9b974d7bc07468600f7e00`
- GPU: NVIDIA GeForce RTX 3060,
  `GPU-458a424a-76e3-65e5-0470-803e0ed131ca`

The static independent audit disposition was
`PASS_LOCAL_SOURCE_MECHANISM__HOLD_PRODUCTION_GLOBAL_PAYLOAD_AND_READS`.
This pilot does not upgrade those production/read holds.

## Boundaries

- one Qwen Up block, not a whole expert, role panel or model panel;
- coordinate descent, not a globally nearest RM6 codeword;
- no current global STRATA execution;
- no outer expert container or cold-read benchmark;
- no literal packet at the selected checkpoint;
- no claim about the project's full-codec F, -10% or -20% milestones.

Verify the sealed evidence with the published manifest digest:

```powershell
& C:\INT2__compression\.tools\python\cpython-3.12.14-windows-x86_64-none\python.exe `
  -I -B research\strata_rm6_qwen_local3060_pilot_v0_20260904\verify_result.py `
  --package research\strata_rm6_qwen_local3060_pilot_v0_20260904 `
  --manifest-sha256 <PUBLISHED_SHA256>
```
