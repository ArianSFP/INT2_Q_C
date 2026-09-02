# V7 independent source-only evidence

Date: 2026-09-02. This is not a manifest/freeze and grants no payload authority.

## Authentication before execution

- Local source: `research/unifilar_wfa_entropy_census_stage0_v7`
- Isolated RunPod copy: `/workspace/uwfa_v7_independent_audit_2cd5e4cc`
- Exact inventory: 17 regular files, 625,600 bytes; see
  `SOURCE_INVENTORY.tsv`.
- Independent UTF-8-ordinal inventory root, domain
  `UWFA-SC-V7-INDEPENDENT-INVENTORY-v1\0`:
  `2cd5e4cc7f53ec0e9ab91f8a9f9a505b8e82e316ab7fe0c1e69e8dcac28bc97f`.
- Local and remote member lengths/SHA-256s matched before Python execution.
- `namei` showed `/`, `/workspace`, and the copied source as real directories,
  not symbolic links.

## Full POSIX suite

Command:

```
cd /workspace/uwfa_v7_independent_audit_2cd5e4cc
/usr/bin/python3.12 -I -B test_source_only.py
```

Result: 66 tests in 78.994 seconds; 65 passed, one expected pre-manifest skip,
zero failures/errors.

The executed names include all new held-bundle, marker mutation/hardlink/link-
branch, literal measurement/alignment, workload telemetry, dependency-component
and all prior source/control/preflight/resource/triplet/E250 regressions.

## Single RTX 5090 source-free receipt

The authenticated v7 script ran once with transaction id
`42424242424242424242424242424242` under
`/workspace/uwfa_v7_gpu_parent_2cd5e4cc/gpu-receipt`.

Receipt facts:

- status: `PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY`;
- `payload_authority_granted=false`, `public_commit_evidence=false`;
- producer source root:
  `334ed49d9cfa3d2772235af3e9da5575b805e494cc41df303c12ca6648982df4`;
- bound preflight:
  `aeb1e0c66e4619d588991d093434bc8999944bbf7fb855a5fc59dee8295baf9d`;
- GPU: NVIDIA GeForce RTX 5090,
  `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722`, PCI
  `00000000:16:00.0`;
- exact 150 cell records with unique ordered selectors `0..149`;
- all-150 status `PASS_ALL_150_CPU_CUPY_EXACT_REPEATED`, 14.7973192650825 s;
- representative status `PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD`;
- representative workload 184,852,206 updates in 28.4172697890317 s,
  composed of 142,804,641 count and 42,047,565 length updates, with 304
  kernels; projected source/control work 2,187.96147019838 s.

Independent verifier `independent_audit.py` imports no producer module. Its
authenticated SHA-256 was
`766217ea5cf6aa7fbe52c36288a7cb4f797b715cb7a9d86ebff3cab934163ddc`.
It reopened the source and receipt only through held directory/member
descriptors, checked exact bytes, seals, canonical roots, sole marker link,
source inventory, exact selectors and canonical GPU identity. It returned
`PASS_INDEPENDENT_SOURCE_AND_GPU_RECEIPT_AUTHENTICATION`.

Committed receipt identity:

- receipt SHA-256:
  `e3302b8eed9b49d9f2e0d2e82cfc65b620d5c471856c72e4acbd888711f410a2`;
- parent marker SHA-256:
  `c3f81339d5fb75f56a66cd6113f3301b168d95e86db90b634afa9c8f670272f6`;
- directory root:
  `504d1b429346d20c6f22808e8fa5f2dd67590de5686fdc2123fc13286f35fdf9`;
- parent commit:
  `81618051250a9bd1e6b9a3d3c5bda3af3f312beecb72375487009e490d47cd1b`;
- completion seal:
  `6fa3fca867787836940a07aa4a9e677a439ed84135d08433206c61befbf63e1b`;
- final directory device/inode: `66307 / 12902063750`;
- marker device/inode/link count: `66307 / 10773449683 / 1`.

Authenticated copies are retained here. Their file hashes are:

- `COMPLETE.authenticated-copy.json`:
  `6adcd9056a42d55511dcb4906758e7f43702c2afcdd35c4e70ac6df7fa6a41a5`;
- `RUN_STATE.authenticated-copy.json`:
  `95044d70ae6fb5db83486d67a4187d6a3e271c31b505ee1acc54674047ed4996`;
- `GPU_DEV_RECEIPT.authenticated-copy.json`: receipt hash above;
- `parent-marker.authenticated-copy.json`: marker hash above.

No Qwen/model payload path was opened.
