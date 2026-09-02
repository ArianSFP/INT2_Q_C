# Independent source audit: ε-TCQ STRATA bound gate v1

Date: 2026-09-02

This is a payload-free audit of source manifest
`e926575ac1a78a85d08e94e63d1cc85d70b1544e5b352b6abc45cb8653d83706`
and source root
`5c3b3a6cb1e2740202710526429a34cca54fcc9105c18820cf6206d276166380`.

The audit separately authenticates the 116,835-byte independent STRATA decoder
at SHA-256
`85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e`.
It parses source constants and exact state-object declarations without opening
or executing a STRATA container.

## Verdict

V1 correctly retracts v0's synthetic six-events-per-coordinate integration.
The authenticated decoder performs six level-major polar SC passes; one output
index is assembled only after six complete polar output planes. V1 has no
coordinate-local candidate API and no four-level replacement fallback.

For the frozen straightforward resumable state representation at `N=2^21`,
beam 32, the independently recomputed peak is exactly:

```text
LR f64/path             176,160,768 bytes
partial sums u8/path     22,020,096 bytes
six planes u8/path       12,582,912 bytes
index state i16/path      4,194,304 bytes
scalar state/path               256 bytes
frontier (32 paths)   6,878,666,752 bytes
u32 backpointers        268,435,456 bytes
total                 7,147,102,208 bytes
```

That exceeds the frozen 4 GiB cap. This proves the package's stated bound for
that representation; it is not a universal lower bound against a future
persistent or recomputing implementation. The correct verdict remains
`HOLD_PRODUCTION_POLAR_LIST_SCALABILITY`.

The audit also verifies mechanically, with hostile source-only tests, that the
six v0 blockers are closed:

- literal packet construction determines bytes;
- scored bytes equal the packet ledger total;
- state/local/permuted gains are derived from bound FP64 artifacts;
- read amplification is derived from literal ranges;
- every fitting and selection input is exactly the outer development set; and
- each matched-control closure is a sealed full receipt whose JSON digest must
  match an external pin, not a set of assertion booleans.

An independent packet decoder performs a canonical byte re-encode. The
source-free CuPy test executes only a deterministic four-value top-k on the
device. No Qwen, current-codec container, or matched-control payload is opened.

## Reproduction

```bash
python3 -I -B independent_audit.py \
  --package /absolute/path/to/research/epsilon_tcq_strata_bound_gate_v1 \
  --decoder-source /absolute/path/to/strata_v2_klt_mixed_independent_auditor_v1.py \
  --cupy
```
