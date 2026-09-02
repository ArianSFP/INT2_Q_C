# TACTIC-CAGE graph/Krylov Qwen oracle result

Date: 2026-09-02

Status: `HARD_KILL_COARSE_GRAPH_NOT_SOURCE_SPECIFIC_0P03_BPW`.

This directory records the completed retry from an exact clean source
closure. The first launch failed before payload access because a prior
source-only test had created an extra `__pycache__` entry in its staging
directory; that failed log and launcher remain preserved separately.

The experiment is a continuous/ideal containment oracle, not a finite codec.
It reads the authenticated lower-rate coarse frame once, buffers it, derives
all graph/order information from those paid bytes, and compares the Qwen
residual with identical-geometry permutation and moment-matched Gaussian
controls. The compact metrics and interpretation are reported below.

## Result

The winning decoder-derived ordering was `coarse_signed_path_dct`.

- Fixed first 384 DCT coefficients captured `0.0939015892002368` of coarse
  residual SSE, producing relative MSE `0.0335031247089882` and an ideal
  rate-equivalent gain of `0.0711301729356835 bpw`.
- An unimplementable free-support top-384 oracle captured
  `0.422096957207252` and reached relative MSE `0.0213680517277365`.
- The optimistic continuous 384-bit Gaussian-waterfill oracle captured
  `0.343389767350314`, reached relative MSE `0.0242782618835445`, and thus
  nominally crossed the distortion target with a `0.30344543088152887 bpw`
  rate-equivalent gain.
- The strongest identical-geometry control obtained
  `0.30328819836395404 bpw`. The Qwen-minus-control excess was therefore only
  `0.00015723251757482348 bpw`, far below the `0.03 bpw` source-specific
  promotion floor. Graph-over-public advantage was likewise only
  `0.00008484503924318607 bpw`.

The apparent target pass is therefore a dimension-allocation/control effect,
not learnable Qwen structure. No finite codec was run or emitted, and this
graph/Krylov family is closed under the experiment's predeclared hard-kill
rule. The result does not license adding its nominal gain to any other oracle.

## Integrity

- `RESULT.json`: `cd0f6f390feecb5d19c2910130a34b18f9aa16013bfa37549ffc15acb1471372`
- `PROVENANCE.json`: `2ece04ce5a3132f71bcf13760f840f5757beaccda8cb70ec68768de08b5eaeee`
- `COMPLETE.json`: `808b296ee2ee602fe940ad2d58c8134c938f1698cfdbcb82ee476ae46a688653`

The initial staged launch exited before any payload access because source-only
tests had created an unmanifested `__pycache__` directory. Its traceback is
preserved as `FIRST_LAUNCH_FAILURE.txt`. The successful retry used a newly
copied exact source closure rather than deleting or ignoring the extra entry.
