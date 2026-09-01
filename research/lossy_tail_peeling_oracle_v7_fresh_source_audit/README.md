# Fresh independent source audit: lossy-tail peeling oracle v7

Status: **BLOCKED_SOURCE_ONLY_RELEASE_CONFORMANCE**.

This namespace is independent of the producer's tests, verifier, CPU receipt,
and artifact receipt. It authenticates launch manifest SHA-256
`3d5bc5ed95071cc45406d0d2906b54f40d32adad0dffc6323b8fa80ca491ed63`
and its exact eleven-member source-stage closure, then records a fresh
adversarial review. Nothing here authorizes runtime calibration, payload/model
access, GPU execution, or production.

The audit returned BLOCK for five release-conformance defects:

1. `SO_PEERCRED` proves which process holds the peer socket, but the claimed
   preflight-source provenance is reduced to a Python executable inode and
   `/proc/<pid>/cmdline`. A process can rewrite its own argv storage, so the
   command-line bytes are not an unforgeable proof that the frozen preflight
   issued the capability.
2. Source/runtime audit `required_status` strings are selected by the later
   authorization rather than pinned by the frozen contract to exact PASS
   values. A sealed BLOCK receipt can therefore be mirrored as the required
   status.
3. Production result creation is a pathname `mkdir` followed by a separate
   pathname `open`, after a long validation/execution window. It holds no
   authenticated output-parent or new-run-root directory descriptor across
   creation, leaving rename/replacement TOCTOU.
4. The runtime contract requires six memory-evidence fields for every one of
   48 cells and all five stable-order adversaries. Stable-order rows omit
   `total_bytes_before_free` and
   `all_per_cell_gpu_arrays_deleted_before_free`.
5. `build_panel` calls `free_all_blocks()` while last GPU-bearing loop locals
   remain live (`masks`, and for controls `x`/`words`), with no synchronization
   or used/total closure assertion.

Other independently checked areas passed source review: exact stage hashes;
duplicate-key/cardinality checks; capability record/EOF/ack ordering; raw
production firewall ordering before NumPy; `-O` rejection; strict read-valid
selection; recursive NaN/+infinity/-infinity rejection; physical rate,
relative-MSE, F, read, boundary and early-kill formulas; matched-control
separation; and source-audit isolation from model, CuPy/CUDA/GPU and network.

The host is Windows and cannot execute Linux `SO_PEERCRED`/`SOCK_SEQPACKET`.
The audit therefore replayed an independent peer/one-record/EOF state model
and inspected the exact source ordering, while explicitly declining to treat
producer Linux tests or receipts as evidence. This environmental limitation
does not cause the BLOCK: every blocker above is established by frozen source
and contract bytes.

## Verify

From this directory in PowerShell 7:

```powershell
pwsh -NoProfile -File .\verify_audit.ps1
```

The verifier rejects audit-artifact or target-stage drift, duplicate JSON
keys, seal mismatch, changed findings, or changed adversarial probes. A valid
run prints one compact JSON object whose status is
`BLOCKED_SOURCE_ONLY_RELEASE_CONFORMANCE` and exits successfully because it
has verified the integrity of a BLOCK receipt—not because the payload passed.

The seal fields in `audit_manifest.json` and `audit_receipt.json` use a simple
zero-slot construction: SHA-256 is computed over the exact UTF-8 file bytes
after replacing only that file's 64-hex seal value with 64 ASCII zeroes. The
audit manifest additionally binds exact byte lengths and SHA-256 values for
the receipt, verifier, README, and all eleven target-stage members.

