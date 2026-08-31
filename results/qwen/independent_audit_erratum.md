# Erratum: `manifest_alignment_ok` comparison bug

Date: 2026-08-31

This erratum applies to `results/qwen/independent_audit.json`. It preserves
the original audit output and documents a derived-field bug rather than
silently rewriting the evidence.

## What was wrong

The independent audit computed `manifest_alignment_ok` using this comparison:

```text
manifest.canonical_block_index
  == summary.blocks_detail[].canonical_block_index
  == decoded.source.block_index
```

The last equality is invalid. In these decode artifacts,
`decoded.source.block_index` is the local index inside the separately
extracted one-block source file. It is `0` for all 32 blocks. The manifest's
`canonical_block_index` is instead the position within the original full
tensor and is nonzero for 21 of the 32 rows.

## Evidence and corrected mapping

- The original audit reports exactly 21 false rows in each variant.
- The manifest contains exactly 21 rows with nonzero canonical block index.
- Those sets coincide exactly.
- `manifest.canonical_block_index` equals the summary canonical index for
  32/32 rows in both variants.
- `manifest.source_local_block_index` equals the decoded source index for
  32/32 rows in both variants.
- Ordinal, ID, role, tensor, source path, and BF16 source hash also match
  32/32 in both variants.

Representative cases include `attention_o.l47.b31` (canonical index 31,
local extracted-file index 0), `embedding.b1186` (canonical 1186, local 0),
and `lm_head.b600` (canonical 600, local 0).

The evidence-backed checks are therefore:

```text
manifest.canonical_block_index == summary.canonical_block_index
manifest.source_local_block_index == decoded.source.block_index
```

## Impact

The scientific conclusions do not change. The bug affects only the derived
`manifest_alignment_ok` field in the independent audit JSON. It does not
affect the frozen manifest or panel hashes, implementation hashes, reservoir
accounting, source-hash checks, SC-seed checks, decode-pass counts, or MSE
recomputation.

The unchanged paired result is:

- exact MSE `0.0631987377412609`, joint gate fail;
- RHT MSE `0.0528944847492712`, joint gate pass; and
- relative RHT improvement `16.3045234133882%`.

