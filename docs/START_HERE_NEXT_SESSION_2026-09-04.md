# Start here: next-session handoff

Pause date: 2026-09-04

Repository: `C:\INT2__compression\INT2_Q_C`

Remote: `https://github.com/ArianSFP/INT2_Q_C`

This is a convenient clean scientific pause.  All payload experiments have
stopped.  The completed work separates several dead memoryless branches from
two still-open multiscale/spatial source-model branches.

## Read these first

1. `docs/HYPERPATH_RG_LOCAL3060_CHECKPOINT_2026-09-04.md`
   — exact results, hashes, decisions, and distance to `F=0.9/F=0.8`.
2. `docs/HYPERPATH_RG_RESEARCH_FRONTIER_2026-09-04.md`
   — architecture rationale, read-bandwidth constraints, and revised order of
   work.
3. `docs/MOSAIC_Q_EXECUTION_CHECKPOINT_2026-09-03.md`
   — prior audited baseline and historical branch table.
4. `docs/INT2_Q_VORPAL_REPOSITORY_AUDIT_2026-09-03.md`
   — why VORPAL's mixed-role number is not a universal MoE checkpoint.

## Current target and best result

The target is a universal SwiGLU-MoE PTQ packet with:

```text
2.15 <= physical rate <= 2.5 bpw
F = pooled relative_MSE * 2^(2R) <= 0.8
maximum routed cold read < 2x, ideally near 1x
```

Best deployable finite checkpoint remains:

```text
R=2.5, D=0.030902167403153148, F=0.9888693569009007
```

No `F=0.9` or `F=0.8` finite checkpoint has been reached.  Do not add gains
from separately fitted experiments.

## Completed local-3060 Qwen evidence

### TETRAPATH fixed labels

Directory:
`research/tetrapath4_fixed_label_qwen_probe_local3060_20260904`

Key files:

- `RESULT.json`
- `README.md`
- `verify_result.py`

Result: maximum raw four-way gain `0.00003870032 bpw`; hard kill.

### TETRAPATH stochastic label-flexible upper aperture

Directory:
`research/tetrapath4_ba_qwen_probe_local3060_20260904`

Key files:

- `RESULT_STAGE3_2PLUS2_ALL8.json` — authoritative all-eight aperture;
- `RESULT_OUTLIER_2PLUS2.json` — deeper 240-iteration outlier check;
- `run_ba_probe.py`
- `verify_result.py`
- `README.md`

Result: largest raw irreducible full-vs-best-2+2 gain
`0.03087325 bpw`; best control-corrected gain `-0.00028168 bpw`.
Irreducible memoryless four-way coding is stopped.

The same result bounds PAIRPATH's same-role cross-expert topology at only
`0.00316455 bpw` maximum.  The one real pair signal is expert-local Up/Down:
mean `0.0316639 bpw`, maximum `0.0521481 bpw`, with approximately 1x natural
expert reads.  It is a clue, not enough for a packet.

### True six-plane STRATA-RM6

Directory:
`research/strata_rm6_qwen_local3060_pilot_v0_20260904`

Key files:

- `RESULT.json`
- `README.md`
- `verify_result.py`
- `SOURCE_MANIFEST.json`

Result: best Qwen reduction `5.23756%`, matched Gaussian `7.07671%`; selected
Qwen stream `2.75 bpw`, and every checkpoint exceeded 2.5 bpw.  The tested
local RM(5,12)^6 bank is stopped.

## Mechanism packages and hostile audits

### PAIRPATH

- Corrected/source-sealed r2:
  `research/pairpath_fl_same_layer_microcodec_v0_20260903_r2`
- r2 hostile audit:
  `research/pairpath_fl_same_layer_microcodec_v0_20260903_r2_independent_hostile_audit_20260904`
- repaired source-only r3:
  `research/pairpath_fl_same_layer_microcodec_v0_20260904_r3`
- local CuPy preflight and its audit:
  `research/pairpath_p2_local3060_cupy_preflight_v0`
  `research/pairpath_p2_local3060_cupy_preflight_v0_independent_audit_20260904`

Do not build a Qwen PAIRPATH payload.  The stronger BA relaxation already
hard-killed the topology.

### TETRAPATH mechanism

- source fixture:
  `research/tetrapath4_source_oracle_v0_20260904`
- hostile audit:
  `research/tetrapath4_source_oracle_v0_20260904_independent_hostile_audit`

The XOR/fiber and 4/3-read mechanism are valid.  The original alternating
optimizer is not a certified oracle; use the BA results above for conclusions.

### RENORM-Q

- v0 source mechanism:
  `research/renorm_q_smallblock_v0_20260904`
- hostile audit:
  `research/renorm_q_smallblock_v0_20260904_independent_hostile_audit`

The exact fixed-law tree DP is valid, but v0 is blocked by non-Kraft NLLs,
caller-controlled descriptor cost, unreachable states, and an incorrect LCB
threshold.  The next session should create a new v1 package, never modify the
audited v0, and obtain a fresh independent audit before Qwen access.

### COCHAIN-Q

- v0 source mechanism:
  `research/cochain_q_plaquette_v0_20260904`
- hostile audit:
  `research/cochain_q_plaquette_v0_20260904_independent_hostile_audit`

The public syndrome-fiber mechanism and 1x logical read topology are valid.
Qwen is blocked by manifest closure/pinning, dtype coercion, and missing
full-expert normalization.  A partial, interrupted repair exists locally at
`research/cochain_q_plaquette_v1_20260904`; it is not sealed or part of the
authoritative checkpoint.  Finish it in a new turn, then audit it independently.

## Partial interrupted work — do not treat as evidence

These directories may exist in the working tree but were interrupted at the
pause and are not authoritative:

- `research/cochain_q_plaquette_v1_20260904`
- `research/tetrapath4_ba_qwen_probe_local3060_20260904_independent_audit`
- `research/strata_rm6_qwen_local3060_pilot_v0_20260904_independent_audit`

Resume or discard them only after inspecting exact contents.  Never infer a
PASS from their presence.

## Local runtime and payload bindings

Use only the local RTX 3060:

```text
GPU UUID: GPU-458a424a-76e3-65e5-0470-803e0ed131ca
CuPy Python: C:\INT2__compression\.venv-cupy\Scripts\python.exe
stdlib Python: C:\INT2__compression\.tools\python\cpython-3.12.14-windows-x86_64-none\python.exe
payload root: C:\INT2__compression
repository: C:\INT2__compression\INT2_Q_C
```

Pinned Qwen panel:
`research/same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3/panel_lock.json`.

CuPy on Windows requires the process-local DLL setup used in
`research/tetrapath4_ba_qwen_probe_local3060_20260904/run_ba_probe.py`.

## Recommended next sequence

1. Repair RENORM-Q into a new v1 with normalized probability/Kraft checks,
   frozen map descriptors, reachable states, and the exact 0.03-bpw LCB rule.
2. Finish COCHAIN-Q v1 closure, then cross-audit both repaired packages.
3. Give only a source-audited survivor a separately named local-3060 Qwen
   capability.
4. For COCHAIN, require one legal six-plane reconstruction and pooled
   original-source SSE; never sum per-plane gains.
5. Prefer expert-local multiscale Up/Down factors: they preserve near-1x reads
   and are the only new positive clue in this checkpoint.
6. Stop any aperture below `0.045 bpw`; demand at least `0.10 bpw` before
   expensive finite integration and approximately `0.18 bpw` before latent or
   stochastic-coding engineering.

There are no running experiments at this pause.
