# ε-TCQ STRATA bound gate v1

Date: 2026-09-02

Status: frozen source-only successor. No Qwen weight, current STRATA/POLARIS
container, or matched-Gaussian payload is accepted by the sealed runner. An
independent source audit is required before any payload path may be added.

## Scientific correction to v0

V0's source-free fixture modeled a 64-way weight index as six independent
arithmetic events at that coordinate. That is not the real STRATA codec.

The current codec processes six complete polar levels. At each level, selected
internal SC decisions occur in level-major causal order. The polar transform
couples one internal decision to multiple output coordinates. Six complete
output bitplanes are finally assembled into each index `0..63`. Consequently,
changing one coordinate index is generally not a legal local transition.

V1 therefore freezes two separate facts:

1. `strata_replay_adapter.py` authenticates the pinned independent decoder,
   selected/frozen internal SC state, causal Q0.16 frequencies, canonical
   arithmetic replay, all six polar output planes, exact 64-index assembly,
   literal current-packet re-encode, and byte-identical FP64 reconstruction.
2. Its coordinate-local choice method always returns the typed hold
   `HOLD_COORDINATE_LOCAL_EPSILON_INVALID_FOR_LEVEL_MAJOR_POLAR_SC`.

There is no direct four-level replacement path in this package.

## The only legal ε-TCQ search unit

A future search must branch on selected internal SC decisions while retaining
the resumable state of the complete six-level polar block. The source-free
`tiny_six_level_oracle` exhaustively demonstrates that unit on bounded blocks.
It is not production evidence.

For a straightforward resumable implementation, V1 charges per beam path:

- the FP64 SC likelihood register;
- the partial-sum register;
- six completed output planes;
- the current 64-way index state;
- arithmetic/WFA state; and
- literal u32 backpointers.

At `N=2^21`, beam 32 already exceeds the frozen 4 GiB cap. Production remains
`HOLD_PRODUCTION_POLAR_LIST_SCALABILITY`. Promotion requires a separately
frozen device-resident resumable kernel, wired CuPy top-k, bounded literal
prefix storage, and deterministic dominance over a host reference. A host
prefix-copy beam is not an allowed fallback.

## V0 audit blockers closed in the bound driver

The generic bound driver is executable on source-free fixtures and closes the
mechanical audit failures without claiming Qwen readiness:

- fixed bytes come only from a literal packet built and independently decoded;
- every scored row's byte count is required to equal the packet ledger total;
- local, state, and state-permuted gains are recomputed from bound FP64 source
  and reconstruction artifacts;
- routed read amplification is recomputed from literal page ranges, with no
  compressed expert second pass;
- topology, frequency, centroid, directory, frame, payload, and padding bytes
  are all conserved;
- every outer component is present exactly once, and topology/frequency/
  centroid fitting plus inner selection use the complete development set;
- each of eight matched controls needs a sealed receipt whose complete JSON
  hash is externally pinned; assertion booleans are not a closure; and
- the control gain is recomputed from its authenticated literal packets and
  score artifacts.

`candidate_packet.py` is a source-gate packet grammar, not a completed STRATA
replacement codec. `independent_decoder.py` implements its parse and canonical
re-encode without importing the builder or driver.

## Source-only checks

```bash
python3 -I -B research/epsilon_tcq_strata_bound_gate_v1/test_source_only.py
python3 -I -B research/epsilon_tcq_strata_bound_gate_v1/verify_source.py \
  --package research/epsilon_tcq_strata_bound_gate_v1
```

The sealed runner has only authorization and source-manifest arguments. Its
output remains a typed hold and grants no payload authority.
