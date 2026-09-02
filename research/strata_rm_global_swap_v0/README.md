# STRATA RM-ordered global frozen-set swap v0

This is a **source-only, no-payload launch gate** for the cheapest honest
LOGIC-Q experiment that can reuse the current STRATA codec.  It changes one
thing only: for every existing global polar block and every one of the six
level-major passes, it replaces the integer-Q31 BEC reliability set by a set
with the **same selected count** `K`, ordered by descending Arikan-generator
row Hamming weight.

For an internal SC phase `i` under the pinned encoder convention

```text
external_u = internal_u[bit_reverse]
x          = polar_transform(external_u)
```

the generated row has weight `2**popcount(i)`.  The candidate therefore orders
internal phases by

```text
(-popcount(i), i)
```

where the ascending phase index is the normative tie break.  It selects the
first `K` phases and freezes the rest.  If `K` equals
`sum(comb(m,j), j=0..r)` for `N=2**m`, the set is exactly `RM(r,m)`; otherwise
it is named **RM-ordered truncated polar**, never exact RM.

## Pinned lineage

The gate authenticates, before import or launch:

- `agent_polaris_qwen_rht_encoder.py`, SHA-256
  `062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0`;
- `bg_codec_bec_encoder.py`, SHA-256
  `456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267`;
- the frozen independent STRATA auditor,
  `strata_v2_klt_mixed_independent_auditor_v1.py`, SHA-256
  `85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e`.

The source package also pins the committed all-14-block overlap receipt.  That
receipt is admitted only as row-set-overlap evidence.  It contains no
candidate reconstruction or codelength and is explicitly forbidden as an RD
result.

## Count-preserving adapter

`swap_adapter.install()` first asks the pinned BEC implementation for its six
current flags.  It counts the selected phases in each returned internal-order
flag, constructs the RM ordering, and verifies equality before replacing the
base hook.  Thus neither a rounded capacity nor a locally copied profile can
silently change `K`.  The intended production block lengths are exactly
`2**20` and `2**21`.

The pinned historical base CLI itself rejects lengths above `2**18`.  The
adapter is therefore an integration component for the current full STRATA
encoder, not permission to call that old CLI on a smaller proxy and transfer
the result.  A payload launch stays held until an independently audited global
encoder installs the hook at the authenticated current flag boundary.

## Cosets

The current format uses decoder-reproducible frozen bits from
`default_rng(sc_seed + 1_000_003 * level)`.  This package supports that mode as
`current_random` only.  A zero-frozen set would expose the direct polynomial
coordinate-function interpretation, but it is a different affine coset and
the pinned upstream API has no honest selector for it.  `zero` therefore
raises `HeldCosetFork`; it may be tested only by a separately versioned and
charged packet fork.  Results from the two cosets must never be pooled.

## Physical-rate boundary

`K`, selected fractions, ideal NLL, and the previous overlap receipt are not
rates.  A production claim may use only the byte length of one literal packet
that includes every expert-local header, profile/order selector, FP scale,
seeds, arithmetic payload, termination, CRC/trailer, padding and charged
shared bytes.  The independent decoder must reconstruct the source-domain
weights, regenerate all causal arithmetic probabilities, consume the packet
exactly, and canonically re-encode byte-for-byte.  `result_contract.py` makes
these conditions machine-checkable but does not itself establish that an
external receipt is truthful; an independent result auditor is mandatory.

## Current disposition

`HOLD_PAYLOAD_PENDING_INDEPENDENT_SOURCE_AUDIT_AND_GLOBAL_PACKET_INTEGRATION`

No Qwen, coarse, control, or model payload is opened by any source-gate or
smoke command.  Passing the tests establishes mechanism and source binding
only; it is not an MSE, rate, or target result.

Source-only commands:

```bash
python test_source_only.py
python hostile_tests.py
python run_source_gate.py --external-root /workspace/INT2__compression
python cupy_order_smoke.py --output cupy_order_smoke_receipt.json
python verify_source.py --package .
```

