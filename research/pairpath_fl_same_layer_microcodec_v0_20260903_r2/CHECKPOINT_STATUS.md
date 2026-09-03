# PAIRPATH-P2 r2 interrupted checkpoint

Status: **UNSEALED SOURCE-ONLY WORK IN PROGRESS — DO NOT AUTHORIZE PAYLOAD OR GPU EXECUTION.**

Work stopped on user request while the final documentation/verifier patch was
being applied. The following prepared source files are present:

- `pairpath_r2_core.py`
- `source_free_fixtures.py`
- `test_source_only.py`
- `run_gate.py` (all execution flags disabled)
- `design_lock.json`
- `README.md`

The package is not sealed. `SOURCE_MANIFEST.json` and `verify_source.py` are
missing. An independent hostile audit has not run. No Qwen payload capability
or local-GPU authority was created. The generated `__pycache__` directory is
not part of the intended source closure and must not be staged.

Last completed checks before the stop:

- `pairpath_r2_core.py` compiled successfully before the final oracle additions.
- Nine synthetic tests subsequently ran; eight passed and one failed only
  because the E=6 expected valid descriptor count was written as 60 instead of
  the correct 45. That test constant was amended to 45, but the suite was not
  rerun after the requested stop.
- A source-only IID literal packet had independently decoded at `7/3 bpw` with
  conservative maximum cold-read amplification approximately `1.27711x`.
- The source-only optimistic oracle killed IID structure and survived the
  deliberately identical-expert positive control in direct probes.

Known scientific anchors frozen in `design_lock.json`:

- real fixed-label CBIB net ideal gain: `0.000010730760043135371 bpw`;
- required Up/Down gain with Gate unchanged: `0.22933495044437174 bpw`;
- optimistic early-kill threshold: `0.045 bpw`;
- physical-engineering margin: `0.27 bpw`.

Before any continuation, rerun syntax/tests, inspect the interrupted files,
add a fail-closed verifier, build a canonical exact-file manifest, and obtain a
fresh independent hostile audit. Any eventual payload capability must be a
separate, pinned, local-RTX3060-only package.
