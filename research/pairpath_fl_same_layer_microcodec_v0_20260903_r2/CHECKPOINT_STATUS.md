# PAIRPATH-P2 r2 sealed source checkpoint

Status: **SEALED SOURCE-ONLY HOLD — INDEPENDENT HOSTILE AUDIT REQUIRED.**

The interrupted package was repaired and its complete ten-test source-only
suite passes.  The three material pre-seal corrections are:

- deterministic symmetric multistarts for independent and joint label search;
- a role-conditioned fixed-label mutual-information ceiling;
- one global Up/Down rate-distortion multiplier across both optimized roles.

The executable package remains payload-blind.  `run_gate.py` has all execution
flags disabled.  No Qwen payload, local GPU, network, RunPod, production runner,
or deployment authority was accessed or created.

The source manifest and verifier bind the exact closure.  This self-seal is not
an independent audit and cannot authorize execution.  The next action is a
hostile audit by a different agent; only a passing audit may lead to a separate
one-use capability restricted to the pinned local RTX 3060.

Known scientific anchors remain:

- fixed-label CBIB net ideal gain: `0.000010730760043135371 bpw`;
- required Up/Down gain with Gate unchanged: `0.22933495044437174 bpw`;
- optimistic early-kill threshold: `0.045 bpw`;
- physical-engineering margin: `0.27 bpw`.
