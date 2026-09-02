# SILT-INT2 source-free mechanism v1 independent audit

Date: 2026-09-02

Auditor verdict: **BLOCK**

Reviewed producer boundary: `research/silt_int2_source_free_mechanism_v1`

Authenticated source root:

```text
d43960c62f57f85d1c7c726fbee4f960303d9bf73ef850f93206967266234640
```

This is an independent source-free audit. The producer directory was not
modified. No manifest, source freeze, canonical result, model packet, Qwen
payload, weight tensor, MSE result, or source-gain claim was created or
authorized.

## Executive decision

The exact authenticated producer suite passes all 15 tests on the provided RTX
5090, and the repairs for the v0 arithmetic, algebra, scalar-bound, nominal
owner-ledger, source-root, and CUDA-equality defects are real. Two new hostile
counterexamples nevertheless prevent a source-free v1 acceptance:

1. The claimed cold-page trace is not the access path of the ordinary routed
   decoder. `decode_expert()` and even `trace_expert_cold_pages()` first run a
   whole-container parser that hashes and zero-checks every expert frame. A
   literal artifact whose reported routed amplification is `5/3` is read and
   validated at `3x` by the ordinary API.
2. A failure after the no-replace rename and parent-directory `fsync` makes the
   context-manager abort path delete all members through the still-open staging
   descriptor. It leaves a visible, empty final directory with no `COMPLETE`
   file. Publication is therefore not complete-or-absent under the producer's
   own post-rename fault hook.

There are also non-blocking-to-this-fixture but mandatory-before-promotion
issues: the telemetry schema differs from the design lock and has fail-open
NVML error paths, the search candidate count is unbounded, and the mechanism is
not a universal SwiGLU-MoE codec. In particular, one canonical `128 x (Gate,
Up, Down)` Qwen-shaped layer has `603,979,776` labels and is rejected by the
format's `268,435,456` total-symbol cap.

## Exact source authentication

The source tree contained exactly the eleven allowlisted regular files. The
independently recomputed length-delimited tree root equals the producer's
reported root.

| Member | SHA-256 |
|---|---|
| `POSTIMPLEMENTATION_REVIEW.md` | `ca2941db9cf2756fb393dcfd65da7b9458c5a826d23bfd608bfe92e0b6df6a8f` |
| `README.md` | `e88a2b9043c1b1baf29a86221bad609c9000de0b4a394e13d2293829145b83ea` |
| `cupy_backend_v1.py` | `c6301d34126e4da3f6bcbd709988a256041da3acf34ad7ed29572ed7d26f57dd` |
| `design_lock.json` | `eda38955c2b80084431282daa03725ff9207d406117e3aebf81a5b5afb16dc21` |
| `independent_decoder_v1.py` | `f30a6e8c071d9751de0f48b93cb3ceb038e81cc1417ab92d89188e1468d90713` |
| `run_synthetic_v1.py` | `8f8fadc88a2a0c6f48e5afcabf33f4c8ad6b87a1457580f06942951f35879602` |
| `safe_publish.py` | `273e312b651a3cc014a52cf4256a352845ccd6cc210a7e7374fa012650a07d64` |
| `silt_v1.py` | `bdb6028540cd6e4b09e171553a74bf0a13c195aa846e1dd483f3a0483d29977a` |
| `source_bootstrap.py` | `bcfd78f64fcc946971b67c9906d2a622aed44b6afabc6043eab17f6b38bae78d` |
| `test_source_only_v1.py` | `583a82900b0f8854797d2c87c60f89327fe4f17142ce3d67c1ec1bc5ac18cdf7` |
| `verify_source_v1.py` | `681409501e1449c48a6f901295420f742b46dc1c556dbaa0e59b1cf81f6d9ae3` |

The audit capsule is
[`hostile_audit_v1.py`](hostile_audit_v1.py), SHA-256
`ecb1c3097def3296c420c669d9121e94f9c37ec71d5dd54723db43a0996e56ee`.
It authenticates the producer root before importing NumPy or any producer
module.

## Authenticated RTX 5090 replay

### Commands

The reviewed tree was copied without alteration to the isolated remote path
shown below. The first command imported nothing from the tree.

```bash
cd /workspace/silt_v1_independent_audit_exact_d43960c6/silt_int2_source_free_mechanism_v1
/usr/bin/python3.12 -I -S -B source_bootstrap.py --print-observed-root
```

Observed stdout:

```text
d43960c62f57f85d1c7c726fbee4f960303d9bf73ef850f93206967266234640
```

The documented workspace venv did not reproduce the run:

```bash
/workspace/int2-cupy-venv/bin/python -I -S -B source_bootstrap.py \
  --expected-root d43960c62f57f85d1c7c726fbee4f960303d9bf73ef850f93206967266234640 \
  --entry verify
```

It failed before tests with:

```text
ModuleNotFoundError: No module named 'pynvml'
```

The producer-compatible, already-existing SILT environment was located at
`/tmp/silt-source-free-v0.GwelaC/.venv`. No package was installed or changed.
The successful replay command was:

```bash
/tmp/silt-source-free-v0.GwelaC/.venv/bin/python -I -S -B source_bootstrap.py \
  --expected-root d43960c62f57f85d1c7c726fbee4f960303d9bf73ef850f93206967266234640 \
  --entry verify
```

### Exact replay receipt summary

| Field | Observation |
|---|---:|
| Status | `PASS` |
| Tests / failures / errors / skips | `15 / 0 / 0 / 0` |
| Elapsed | `11.759377353009768 s` |
| Interpreter | `/tmp/silt-source-free-v0.GwelaC/.venv/bin/python` |
| Interpreter SHA-256 | `1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5` |
| Python / NumPy | `3.12.3 / 2.1.2` |
| CuPy / nvidia-ml-py | `14.2.0 / 13.610.43` |
| GPU | `NVIDIA GeForce RTX 5090` |
| UUID | `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722` |
| CUDA logical / NVML physical | `0 / 0` |
| CUDA PCI / NVML PCI | `0000:16:00.0 / 00000000:16:00.0` |
| Compute capability | `120` |
| Driver / CUDA runtime | `580.126.09 / 13020` |
| H2D / D2H logical bytes | `305,832 / 2,483,200` |
| H2D / kernel / D2H CUDA-event time | `3.8109119459986687 / 102.47039914131165 / 1.5322879943996668 ms` |
| CPU reference score / search wall | `7,276.871087728068 / 7,678.491642931476 ms` |
| Host RSS baseline / peak / delta | `335,024,128 / 392,298,496 / 57,274,368 B` |
| Process VRAM baseline / peak / delta | `522,190,848 / 528,482,304 / 6,291,456 B` |
| Device VRAM baseline / peak / delta | `1,053,949,952 / 1,060,241,408 / 6,291,456 B` |
| CuPy pool peak delta | `1,879,040 B` |
| Resource samples | `190` at `2 ms` |

The replay confirms the producer's finite source-only tests and the exact
logical transfer-byte ledger. It does not validate a Qwen or other model
source-law claim.

## Blocking finding 1: the reported cold trace is not the decoder trace

### Static cause

The ordinary route is:

```text
decode_expert(packet, e)
  -> parse_container(packet)
       -> for every directory entry:
            parse_frame_header(all pages of that frame)
              -> SHA-256(frame body)
              -> scan the complete page-alignment zero tail
  -> decode only frame e
```

The relevant implementation locations are:

- `silt_v1.py:1294-1296`: `decode_expert()` calls `parse_container()` first.
- `silt_v1.py:1261-1277`: `parse_container()` loops over every expert frame.
- `silt_v1.py:901-907`: each frame parser reads/hashes its body and scans its
  page tail.
- `silt_v1.py:1365-1366`: even `trace_expert_cold_pages()` performs that full
  parse before constructing its synthetic range list.

The `InstrumentedPageReader` therefore traces an intended list assembled after
the real validation accesses. It does not wrap or observe the ordinary decode.
The independent decoder is even more explicit: its `parse_container()` creates
a fully parsed `Frame` for every entry before any selected reconstruction.

### Exact finite counterexample

The audit built a three-expert, equal-frame, finite GF(2) artifact using only
the source generator. Expert 0 was selected.

| Quantity | Exact value |
|---|---:|
| Global/directory/model region `G` | `12,288 B` = pages `0,1,2` |
| Each local frame `F_e` | `8,192 B` = 2 pages |
| Literal container | `36,864 B` = 9 pages |
| Owner share of expert 0, `G/E + F_0` | `12,288 B` |
| Claimed selected pages | `0,1,2,3,4` |
| Claimed cold bytes, `G + F_0` | `20,480 B` |
| Claimed amplification | `20,480 / 12,288 = 5/3 = 1.6666666666666667x` |
| Ordinary parser pages | `0,1,2,3,4,5,6,7,8` |
| Ordinary materialized/touched bytes | `36,864 B` |
| Actual amplification | `36,864 / 12,288 = 3x` |

Owner conservation itself is correct:

```text
3 * 12,288 owner bytes = 36,864 literal bytes.
```

The flaw is solely the numerator: the ordinary path consumes the whole object,
not `G + F_e`.

For a functional access-dependency proof, the audit flipped byte offset
`36,863`, on page 8 in unselected expert 2. Bytes in pages `0..4`—the entire
claimed read union for expert 0—remained byte-identical. Both ordinary calls
nevertheless rejected:

```json
{
  "decode_expert": "frame page zero tail",
  "trace_expert_cold_pages": "frame page zero tail"
}
```

Thus an unclaimed page is observably consumed by both the decoder and the
supposed trace function.

### Required repair

A v2 acceptance needs a literal routed file/descriptor decoder, not a function
whose input is an already materialized whole-container `bytes` object. It must:

1. read and authenticate the global header, page-aligned directory, model, and
   only the selected expert frame;
2. never parse, hash, or zero-scan another expert frame on a cold route;
3. instrument the actual file-range reads made by that decoder;
4. independently decode/re-encode the selected frame through a separately
   implemented ranged reader; and
5. distinguish any optional whole-artifact installation audit from the strict
   cold route. A prior warm validation cannot be used for the frozen cold gate.

## Blocking finding 2: post-rename cleanup destroys the final tree

### Static cause

`SafePublisher.finish()` performs the no-replace rename, `fsync`s the parent,
then invokes the `published_and_parent_fsynced` fault checkpoint while
`self.finished` is still false and `self.staging_fd` still points to the renamed
directory. If that checkpoint or the later final-tree rehash raises,
`__exit__()` calls `abort()`.

`abort()` enumerates and unlinks files through `self.staging_fd`. File
descriptors follow the inode across rename, so those are now the visible final
files. It then tries to remove the old hidden staging name, which no longer
exists.

### Exact filesystem evidence

The independent publisher subclass only observed the filesystem at the
existing producer checkpoint and then delegated back to the unmodified
producer code. At the checkpoint immediately after rename and parent `fsync`:

```json
{
  "stage": "published_and_parent_fsynced",
  "parent_members": ["postrename"],
  "final_members": ["ARTIFACTS.json", "COMPLETE", "a.bin"],
  "complete_present": true,
  "final_inode": 12903102082,
  "staging_fd_inode": 12903102082,
  "same_inode_after_rename": true
}
```

After the injected `PublicationError` entered `__exit__()`, the externally
visible state was:

```json
{
  "final_path_visible": true,
  "final_members": [],
  "complete_present": false,
  "exception": "FileNotFoundError: [Errno 2] No such file or directory: '.postrename.staging.262857.f335f3b9229c15389c1af2f9'"
}
```

The second exception masks the injected failure. The final path is neither
absent nor complete.

A separate constructor fault at `staging_created` left one hidden orphan:

```text
.constructor.staging.262857.561825b1d99a204b7ab67ded
```

That constructor case does fail closed with respect to the final name, but it
leaks a directory and descriptor until process teardown.

### Required repair

The publisher needs an explicit `STAGING -> COMMITTED -> VERIFIED` state
machine. Immediately after a successful rename and parent `fsync`, it must mark
the object committed. `__exit__()` must never call staging cleanup after that
transition. A post-commit verification failure may make the operation report a
failure, but it must not turn a complete atomically published tree into a
partial final tree. Constructor exceptions must close the staging descriptor
and remove the hidden directory. Hostile tests must inject at every checkpoint,
including `staging_created`, `published_and_parent_fsynced`, and each
post-publication rehash read.

## Positive independent checks

These parts of the v1 repair survived independent checks:

| Gate | Independent observation |
|---|---|
| Owner denominator | The historical `G=8192`, `F=[4096,8192,...]` expert-0 case is exactly `12/5`; owner shares conserve literal bytes. |
| Strict boundary | Four equal 4 KiB frames with `G=8192` are exactly `2/1` and all fail strict `<2`. |
| Expert bounds | `E=0`, `257`, and `2^32-1` reject in `1.81–2.73 us`; lane `2049` rejects before factorial in `4.60 us` on the replay host. |
| Arithmetic | 320 independently generated Q16 cases and 81,950 symbols matched producer/independent encoders and exhausted the declared meaningful bits exactly. |
| Canonicality mutations | `meaningful_bits-1`, nonzero guard, and valid-hash GF(2) selector ID 6 were rejected by both decoders. |
| GF(2) maps | IDs `0..5` are six distinct bijections; IDs 6 and 7 reject. |
| Z4 maps | IDs `0..7` are eight distinct bijections. |
| Root hostility | Extra source member and symlinked member both reject before import. |
| Runtime fixtures | Producer's authenticated replay passed `E=1,128,249,250,256` with its small unequal geometries. |
| GPU equality | Producer's selected CPU/CuPy coefficients and CuPy inverse matched exactly on RTX 5090. |

## Telemetry and resource-bound discrepancies

These did not falsify the successful canonical RTX receipt, but they prevent a
claim that the telemetry implementation is generally fail-closed:

1. `design_lock.json` requires keys named `vram_baseline_used_bytes`,
   `vram_peak_used_bytes`, and `vram_delta_bytes`. The runtime emits only the
   more specific `vram_device_*` and `vram_process_*` names. The test was changed
   to require the latter, so it does not enforce the literal design lock.
2. `_process_vram()` catches any `pynvml.NVMLError` and returns zero. A mandatory
   process-VRAM measurement can therefore become a plausible-looking zero
   rather than failing closed.
3. Sampling occurs in a daemon thread and exceptions in that thread are not
   stored or re-raised by `finish()`. A sampler that dies after its first sample
   can still return stale peaks.
4. `search_metadata_cupy()` materializes all candidate seeds and retains four
   host arrays per candidate, but imposes no candidate-count or seed-width cap.
   The synthetic runner passes eight trusted candidates; the public source API
   itself does not have the advertised bounded-work property.
5. Reproduction currently depends on an unrelated temporary v0-named venv.
   The normal workspace venv lacks `pynvml`, and the source tree has no locked
   dependency capsule. The successful receipt remains valid, but rebuilding
   its environment is not specified.

Recommended source-level repair is to align the design-lock and receipt names,
make all NVML and sampler failures fatal, cap candidates before list/cache
construction, and publish a dependency lock or explicitly authenticated runtime
capsule.

## Universal SwiGLU-MoE boundary

This producer correctly labels itself a finite-label mechanism, not a model
codec. The audit confirms the following absent pieces:

- no float-to-label PTQ quantizer;
- no canonical Gate/Up/Down triplet adapter or reconstruction boundary;
- no tensor-tail or role/shape conformance fixture;
- no original-weight reconstruction or raw-MSE scorer;
- no physical `2.15–2.5 bpw` packet; and
- no held-out SwiGLU-MoE evidence.

There is also a concrete capacity mismatch. For the repository's documented
Qwen evaluation geometry, one expert has:

```text
3 * 768 * 2048 = 4,718,592 labels.
```

For 128 experts in one layer:

```text
128 * 4,718,592 = 603,979,776 labels.
```

The format hard cap is `268,435,456` total symbols, so a single v1 container
cannot represent this legal and central geometry even though `E=128` by itself
was runtime-tested. Splitting a layer across containers is not specified; it
would need a universal partition rule, charged metadata, repeated shared-page
accounting, exact triplet reconstruction, and a new cold ledger.

This shape calculation uses only the public repository contract. No Qwen
checkpoint or payload was accessed.

## Independent hostile command and receipt

The exact source-free hostile replay command was:

```bash
cd /workspace/silt_v1_independent_audit_exact_d43960c6/audit
/tmp/silt-source-free-v0.GwelaC/.venv/bin/python -I -B -m py_compile hostile_audit_v1.py
/tmp/silt-source-free-v0.GwelaC/.venv/bin/python -I -B hostile_audit_v1.py \
  --producer-dir /workspace/silt_v1_independent_audit_exact_d43960c6/silt_int2_source_free_mechanism_v1 \
  --expected-root d43960c62f57f85d1c7c726fbee4f960303d9bf73ef850f93206967266234640
```

The audit intentionally returns nonzero with schema
`silt-v1-independent-hostile-audit-receipt` and status `BLOCK`. Its decisive
receipt fields were:

```json
{
  "authenticated_source_root": "d43960c62f57f85d1c7c726fbee4f960303d9bf73ef850f93206967266234640",
  "status": "BLOCK",
  "positive_checks": {
    "arithmetic_fuzz": {"cases": 320, "symbols": 81950, "status": "PASS"},
    "bounds_probe": {"status": "PASS"},
    "canonicality_mutations": {"status": "PASS_REJECTED_ALL"},
    "owner_ledger": {"status": "PASS"},
    "root_hostility": {"status": "PASS_REJECTED_ALL"},
    "selector_exhaustion": {"status": "PASS"}
  },
  "blocking_counterexamples": {
    "cold_path": {
      "claimed_cold_bytes": 20480,
      "actual_ordinary_parse_bytes": 36864,
      "owner_share_bytes": 12288,
      "claimed_cold_amplification": "5/3",
      "actual_full_bytes_api_amplification": "3/1",
      "status": "BLOCK_CONFIRMED"
    },
    "publication": {
      "postrename_fault": {
        "durable_final_members_before_cleanup": ["ARTIFACTS.json", "COMPLETE", "a.bin"],
        "visible_final_members_after_cleanup": [],
        "status": "BLOCK_CONFIRMED"
      }
    }
  },
  "payload_accessed": false,
  "qwen_or_model_accessed": false,
  "manifest_created": false,
  "result_frozen": false,
  "source_gain_claim": false
}
```

## Final authority boundary

SILT v1 at root `d43960…` is not accepted for payload access, Qwen testing,
codec promotion, or a `<2x` routed-read claim. The verified positive components
may be carried into a format-breaking v2, but both blocking counterexamples
must be repaired and independently replayed first.

