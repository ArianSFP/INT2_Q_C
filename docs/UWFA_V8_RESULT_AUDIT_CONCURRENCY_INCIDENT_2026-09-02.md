# UWFA-SC v8 result-audit concurrency incident

Date: 2026-09-02

## Disposition

The first independent replay of the completed v8 real-Qwen early-gate
publication exited nonzero with:

```text
FAIL_QWEN_EARLY_GATE_RESULT_AUDIT: ResultAuditError:
early-gate runner parent: held directory changed [1]
```

This is a failed audit attempt. It is neither a successful integrity audit nor
positive/negative evidence about the WFA source law.

## Why this failure is attributable to concurrent sibling creation

`RetainedDirectory` opens every component of an absolute path and stores a
seven-field identity containing device, inode, mode, size, mtime, ctime and
link count. For the early-gate runner path, component index 1 is
`/workspace`. Any sibling creation directly under `/workspace` changes that
ancestor's mtime/ctime even when the held runner directory, runner file and
every name-to-inode binding remain unchanged.

The failed audit log was created at `2026-09-02 10:15:59 UTC`. While its
descriptors remained open, the separately authenticated v9 primary run was
started at `2026-09-02 10:57:23 UTC`; its detached launcher created log/control
files directly under `/workspace`. The audit then failed its broad ancestor
metadata-stability check at `2026-09-02 11:12:58 UTC`. The held runner
directory itself retained the earlier mtime/ctime
`2026-09-02 09:04:43 UTC` at inspection.

This chronology explains the exact index-1 failure. It does not waive any
content, inode, rebinding or canonical-decode check inside the auditor.

## Required repair

Do not reinterpret or patch the historical failure into a pass. Rerun the
unchanged auditor only after the active v9 job and all other writers to the
shared `/workspace` parent have stopped. A distinct future auditor may narrow
ancestor stability to device/inode/mode/link identity while retaining full
metadata stability for the actual package and publication directories, but it
must be separately source-reviewed before use.

The v8 publication remains nonpromoting even after a successful replay: it
decoded Qwen and stopped before WFA fitting because of its declared runtime
gate.
