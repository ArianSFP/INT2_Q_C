# TETRAPATH-4 independent hostile audit

This package audits the exact unsealed source-only target whose core SHA-256 is
`b303d9d87659d0ae36687fed9ab82b00e1eea8a6bd94ea4769453e42b5fb611a`.

Run from the repository root:

```powershell
python -I -B research\tetrapath4_source_oracle_v0_20260904_independent_hostile_audit\verify_audit.py
```

The verdict is fail-closed for hard-kill and promotion authority, while
retaining the valid source-only XOR/fiber mechanism evidence.  This audit does
not access Qwen, a GPU, a network, or any payload.
