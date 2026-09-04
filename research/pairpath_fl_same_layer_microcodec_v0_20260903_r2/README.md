# PAIRPATH-P2 r2 executable source-only microcodec

## Outcome and boundary

This package closes the PAIRPATH r1 implementation blockers with the smallest
decisive experiment: a two-expert, same-layer, label-flexible joint source
oracle and a literal finite packet. It is payload-blind and contains no Qwen
locator, model hash, GPU code, network code, production runner, or execution
authority. It is not a Qwen result and cannot promote itself.

The prior real Qwen CBIB-1 fixed-label result is already a hard negative: its
best net ideal gain was `0.000010730760043135371 bpw`. The standalone Up/Down
requirement, when Gate remains unchanged, is `0.22933495044437174 bpw`.
PAIRPATH-P2 therefore changes the labels rather than trying to recode the
frozen nearest labels.

## First and cheapest gate

`optimistic_single_letter_joint_gate()` gives the pair source-derived
single-letter distributions and convexified time sharing for free. It fits
equally flexible independent and full 16-state joint label assignments, then
compares the lower rate-distortion envelopes using

```text
G_eq,UD = R_ind - R_pair + 0.5*log2(D_ind/D_pair).
```

The nearest-label mutual-information screen also reports the necessary
fixed-assignment ceiling `I(A;B)/2`; the standalone fixed assignment needs
`I(A;B) >= 0.4586699008887435` bits per coordinate pair before overhead.

- hard-kill below `0.045 bpw`;
- standalone ideal threshold `0.22933495044437174 bpw`;
- physical-engineering margin `0.27 bpw`.

This oracle can kill but never promote because its probability tables and
time-sharing schedule are free.

## Literal finite experiment

For each exact rational lambda and candidate, the executable pipeline:

1. reads canonical finite FP64 `[2,3,N]` values;
2. recomputes binary16 block-RMS scales and all nearest labels;
3. cross-fits eight folds at 2,048-coordinate block granularity;
4. jointly searches the two legal labels and a decoder-visible binary state
   over all `2*4*4=32` choices per coordinate;
5. serializes every scale, count model, state, label, selector and valid-bit
   length using canonical JSON plus exact canonical-prefix streams;
6. independently parses and decodes the packet;
7. scores literal original-source SSE, physical rate and cold-read traffic.

`pair_k2_fixed` is the operational CBIB-like strict nested candidate on the
same scales and nearest labels. `pair_k2_flexible` differs only by allowing
the 16 joint label choices. `independent_fixed` is always run on the identical
source and prevents a pair method from winning merely by omitting a baseline.
Every candidate reports the explicitly typed
`relative_MSE + lambda*literal_bpw` objective. The across-lambda final selector
minimizes literal `F`, then bytes and frozen tie order; its identity and lambda
are in the packet header.

The physical layout is one page-rounded common segment followed by one
page-rounded private segment per expert. Decoding expert `e` touches only the
common segment and `P_e`. Both padded-ownership and conservative raw-ownership
amplification must be strictly below 2. No inter-route cache-reuse credit is
used.

## Blocker closure

- The bounded aperture is a complete function, not disconnected primitives.
- State `0 <= s < 2` is checked at fit, encode and decode.
- The source binding opens actual caller bytes and recomputes source, packet,
  decoded-label and decoded-scale hashes. Invented strings cannot pass.
- Negative/nonfinite SSE or nonpositive energy is rejected.
- The general pair-first descriptor now materializes its exact binary tree;
  all small descriptor words are exhaustively replayed in the KATs.
- Affine controls transform continuous source values, then recompute scales and
  labels and rerun the complete selector. The Gaussian control does the same.
- Joint-block bootstrap and both whole-expert and whole-layer refit diagnostics
  are executable.
- The boundary KAT shows flexible labels can make the pair candidate physical
  where the fixed pair loses, while the independent candidate still prevents
  a false breakthrough. IID and perfectly aligned oracle KATs fail and survive
  respectively.

## Required next payload package

Only after an independent hostile source audit should a separate capability be
built. It must be local-RTX3060-only and pin the GPU UUID, Python executable,
complete wheel closure, exact Qwen revision, exact layer/expert/file hashes,
one-use claim and result destination. It must have no RunPod or remote path.

The payload sequence is fail-fast:

1. run only the fixed-assignment MI and optimistic joint oracle on public,
   source-closed Up/Down panels;
2. stop immediately if `G_eq,UD < 0.045 bpw`;
3. run iid, covariance-matched and within-stratum value-permutation complete
   refits only after the source gate survives;
4. construct the finite packet only at or above the standalone threshold, or
   if a predeclared nesting proves a smaller gain sufficient;
5. promotion still requires two whole held-out layers, all bytes, one decode,
   `F<=0.8`, and maximum cold read below 2.

## Sealed source-only status

The interrupted checkpoint was reviewed before sealing.  The equal-flexibility
oracle now uses the same deterministic multistart bank for its independent and
joint models, computes the frozen-label ceiling conditional on decoder-visible
role, and uses one global Up/Down rate-distortion multiplier.  Ten source-only
tests pass.  The exact source closure is bound by `SOURCE_MANIFEST.json` and a
fail-closed verifier.

This remains a **source-only hold**.  It grants no Qwen, GPU, payload,
deployment, or result authority.  A different agent must still perform a
hostile source audit before a separately named, local-RTX-3060-only one-use
payload capability may be considered.

## Verification

From the repository root, compute rather than trust the manifest digest:

```powershell
$p = "research\pairpath_fl_same_layer_microcodec_v0_20260903_r2"
$m = (Get-FileHash -Algorithm SHA256 -LiteralPath "$p\SOURCE_MANIFEST.json").Hash.ToLowerInvariant()
python -B "$p\verify_source.py" --package "$p" --manifest-sha256 $m --self-test
```
