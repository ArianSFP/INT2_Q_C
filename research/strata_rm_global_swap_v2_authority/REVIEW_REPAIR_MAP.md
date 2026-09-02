# Frozen v1 review to v2 repair map

| v1 review finding | v2 source closure |
|---|---|
| Root link resolved before rejection | `real_directory` and `verify_source.py` reject the original root via `lstat` before resolution. |
| External source hash-to-import race | `snapshot_pinned_files` creates an exact read-only private closure; `current_snapshot_worker.py` imports only from it and the parent rehashes it after use. |
| Decoder hash-to-exec race | `_run_case` snapshots exact decoder and launcher bytes, executes those copies, then rehashes them. |
| Decoder audit self-declared | `authenticate_decoder_audit_capability` requires separately passed manifest, source-root, receipt, worker, and launcher hashes plus an executed PASS receipt. |
| Decoder-reported reads trusted | `instrumented_decoder_worker.py` wraps packet reads, records byte intervals, supplies no source paths, and blocks explicit non-packet reads and process/network escape. |
| Model/control/family declarative | `authenticate_scientific_capability` is a separately pinned auditor object; the experiment commitment holds only capability IDs and packet pins. |
| Cross-family target not enforced | `evaluate_family_acceptance` applies rate, F, and read gates independently to each exact auditor-declared family. |
| Controls did not affect acceptance | The same function subtracts the strongest complete control pool and requires at least 0.03 bpw of source-specific advantage. |
| Hook/parity not independently exercised | The current snapshot worker invokes both global lengths; the parity worker uses independent CPU/GPU algorithms and compares them with frozen v1. |

These are source mechanisms until the frozen tests and optional CuPy workers
produce authenticated receipts.  They do not authorize payload access.
