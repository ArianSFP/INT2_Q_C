# Independent result audit: lossy-tail v8

Status: **PASS_V8_INDEPENDENT_RESULT_AUDIT**.

This standard-library-only audit authenticates the exact one-shot auxiliary
Qwen result and authorization:

- `result.json`: 3,800,771 bytes, SHA-256
  `2f3ebe509fa3c78c2caf6084510bb14e9e2a2fef9cabbdb4b99c9b396a4bfdf9`;
- `authorization.json`: 5,960 bytes, SHA-256
  `16bf378c1c6baa23eaff7054ca4c1b82fa06ec45bd89e152956d6a51c752d1ef`.

The verifier rejects duplicate JSON keys and nonfinite trees, recomputes both
canonical internal seals, binds the exact frozen producer/source/runtime-audit
chain, and independently checks 207,847 scientific/evidence assertions before
the audit's self-seal checks. It traverses all 225 serialized source/control
score rows and all 2,700 embedded profile rows, rather than sampling them.

For every score row it reconstructs physical capacity/rate, support and symbol
charges, literal FP16 codebooks, residual allocation RD equations, expert frame
padding, full-container bit closure, and logical/page reads. Across all 225
rows, the independently reproduced maxima are:

- logical read amplification: `1.015752828100462x`;
- page read amplification: `1.022286902319691x`.

Both are strictly below the frozen exclusive `2x` limit. The audit also checks
all 48 matched-control moment cells, all 18 calibrated rows, 12 source receipts,
30 panel memory-release rows, and independently selects and executes the finite
decision tree without trusting the producer's `decision` object.

The recomputed status is **EARLY_KILL_FAR_SHORT**. The best optimistic envelope
is the non-materializable zero-tail-error/raw-adaptive row at 2.5 bpw, but its
matched-control excess is only `0.011386295964214366` s-bpw versus the required
`0.16096404744368115`. The best finite joint score is
`-0.0016032826638503746` s-bpw. A finite residual codec is therefore not
warranted by this bounded scalar-tail family.

## Evidence boundary

The audit deliberately opened no model payload and submitted no GPU work. It
authenticates the twelve result source receipts against frozen bindings, but
does not regenerate source moments or candidate-ledger preimages. The result
does not serialize every search trial, the transient one-use capability record,
a replayable mount namespace, or a kernel file-open trace. Those limitations
are recorded explicitly in `audit_receipt.json`; they do not weaken the pass on
the exact supplied result's integrity and arithmetic, but they prohibit claims
of independent payload reproduction, exhaustive-search replay from the result
alone, or kernel-observed production access counts.

This PASS is an authenticated bounded-oracle early kill, not a finite codec,
broader architecture converse, compression promotion, or new production
authorization.

## Verify

From this directory with Python 3.9 or newer:

```text
python3 -B -I verify_result_audit.py
```

The verifier uses only the Python standard library and requires the exact
frozen producer, source-audit, and runtime-audit directories to remain adjacent.
