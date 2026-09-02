# Independent source audit: TACTIC Ramanujan-384 v0

Date: 2026-09-02

Audited producer:
`research/tactic_ramanujan384_adapter_v0`

Pinned producer manifest SHA-256:
`287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde`

Pinned producer root SHA-256:
`2a66a5d745fc0a31e311cf6ab5f44836726ae341db977bca8eac314df61124ad`

## Disposition

`HOLD_PAYLOAD_AUTHORITY_PENDING_LITERAL_WEIGHT_REPLAY_AND_BACKEND_STABLE_CONTROLS`

The frozen source contains a coherent 48-byte refinement packet, exact
Qwen-shape byte ledger, source-first absolute distortion gate, and explicit
authenticated-input capability surface.  It does **not** yet provide an
end-to-end decoded-composite reconstruction score: `adapter.py` validates the
container and its packet syntax, but the reported MSE is accumulated from
encoder-side corrections before the emitted composite is decoded.  The
adapter never reconstructs weights from `decoded_container["coarse_payload"]`
and `decoded_container["fine_payload"]` and never compares that replay with
the scored reconstruction.

The eight Gaussian controls are also not CPU/CuPy-identical by construction.
The pinned parent calls `xp.random.RandomState(seed)`, and NumPy and CuPy do
not define one shared byte-exact random stream.  Consequently the same seed
does not seal the same control payload across backends.  The production run
can still be made meaningful by freezing one implementation and device, but
it is not a CPU/CuPy reproducibility result.  Prefer a small, explicitly
specified counter-based generator whose binary64 samples are generated once
and replayed identically on both backends.

The producer freeze also has an immediately reproducible closure defect.  Its
manifest records root `2a66...`, which is the SHA-256 of rows serialized in
insertion order (`name,bytes,sha256`).  Its own `verify_source.py` calls
`json.dumps(..., sort_keys=True)`, which serializes those keys as
`bytes,name,sha256` and obtains
`64669f3eeb9dd4f34a9fa36c9c6db592dcf5e37bdeb5ce149b3dbd51e2e24733`.
Thus the frozen verifier must reject the frozen manifest before any runtime
test.  Both digests and the mismatch are pinned by this audit.

No Qwen, coarse, matched-control, or other model payload was accessed by this
audit.  No network access is part of the audit package.  Runtime execution is
explicitly pending in a Python/NumPy/CuPy environment.

## Source-level findings

### Verified mechanics

* The period list is exactly all 120 integers in `[3,127]` that are not powers
  of two.  The coverage-first label construction emits 384 unique
  `(period,shift)` identities and visits every period.
* The independent runtime test proves, over a large prime, that the first
  `phi(p)` shifts of the exact integer Ramanujan sum have rank `phi(p)` for
  every one of the 120 periods.  It also checks each selected atom has exact
  fundamental period and that the literal 4,096-by-384 dictionary has unique
  integer columns.
* The packet is exactly 48 bytes: 42 fixed header bits, at most fourteen
  20-bit `(atom,coefficient)` entries, canonical zero padding, and a 32-bit
  CRC.  Coefficients are nonzero signed 11-bit integers in `[-1023,1023]`,
  with one positive finite binary16 scale for nonzero rank.
* The Qwen geometry has 4,718,592 weights, 1,414,656 coarse bytes, 55,296 fine
  bytes, a 512-byte header, no final page slack, and 1,470,464 physical bytes:
  `R = 359/144 = 2.493055555555555... bpw`.  The payload-only rate is
  `319/128 = 2.4921875 bpw`.
* One contiguous page-aligned object is sufficient for one external fetch.
  That proves a layout upper bound of one object byte per stored object byte;
  it is not a measured storage/HBM trace.  The current `1.0` field is ledger
  arithmetic, because no inference reader is implemented here.
* The absolute `D <= 0.025` gate precedes the phase and eight-Gaussian
  controls.  Each permitted control calls the complete finite packet search.
* Source, coarse artifact, coarse reconstruction, independent receipt, and
  binding bytes are all content-hash checked through explicit caller-supplied
  paths.  Duplicate JSON keys, symbolic links, hard links, identity drift,
  noncanonical role order, packet/container corruption, and padding aliases
  have fail-closed checks.

### Authority gaps

1. **No decoded-weight replay.**  `decode_composite` checks the container but
   returns bytes only.  `run_authenticated_expert` never decodes the coarse
   payload, never applies fine packets taken from the decoded container, and
   never recomputes source-domain MSE from those bytes.
2. **Winner selection is not literally packet-replayed for every candidate.**
   Each candidate rank is scored before support sorting and packet emission;
   only the selected candidate is decoded and re-scored.  Floating-point
   accumulation order can therefore make the design-lock phrase “winner
   metric ... after literal packet decode” stronger than the code.  The final
   selected correction is packet-replayed, so this is a narrow determinism
   gap rather than evidence of a material MSE error.
3. **Backend-dependent controls.**  NumPy and CuPy Gaussian samples are not
   one specified stream, and linear solves can also cross a quantizer boundary.
   The supplied CUDA audit runner records packet/control equality instead of
   assuming it.
4. **Authentication is capability-based, not self-authenticating.**  The
   expected binding digest is the external trust root.  The binding contains
   auditor/input-manifest digest strings, but the actual input manifest and
   auditor source manifest are not supplied or opened by `authenticate_role`.
5. **Universal geometry is conditional.**  The container rejects shapes for
   which the exact `307/128` coarse byte count is nonintegral.  Role tails are
   charged and deliberately ineligible in the ledger; tail scoring also
   includes zero-padded nonexistent coordinates, which is conservative but
   not an exact score over only original coordinates.
6. **Read amplification is unmeasured.**  Contiguous minimal layout supports a
   one-read implementation, but the adapter operates on in-memory bytes and
   does not establish an actual external-storage or HBM trace.
7. **Producer source-root closure fails its own verifier.**  The manifest root
   used insertion-order JSON while the verifier uses sorted-key JSON.  This is
   not file drift: all thirteen member byte counts and SHA-256 values match.

## Commands

Source-only runtime audit:

```bash
python -I -B research/tactic_ramanujan384_adapter_v0_independent_source_audit_20260902/test_independent_source.py
python -I -B research/tactic_ramanujan384_adapter_v0_independent_source_audit_20260902/verify_audit.py \
  --package research/tactic_ramanujan384_adapter_v0_independent_source_audit_20260902 \
  --manifest-sha256 <audit-manifest-sha256>
```

Source-free CPU/CuPy identity audit on the provided GPU host:

```bash
python -I -B research/tactic_ramanujan384_adapter_v0_independent_source_audit_20260902/run_cupy_identity_audit.py \
  --authorization AUDIT_SOURCE_FREE_TACTIC_RAMANUJAN384_CPU_CUPY_IDENTITY_V0 \
  --producer-manifest-sha256 287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde
```

The CUDA command has no payload argument or path-discovery surface.
