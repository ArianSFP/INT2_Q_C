# TACTIC-DH384 finite v3

Date: 2026-09-02

Status: **frozen source-only finite codec awaiting independent source review**.
This sibling repairs the finite-code blockers found in
`tactic_conditional_dyadic_coset_v2`; v2 and
`tactic_actual_coarse_n18_v6` remain unchanged. No Qwen/model payload, live v6
result, RunPod, CUDA context, or network was accessed while this package was
built.

## What is now literal

Every complete 4,096-weight block has one 48-byte record:

```text
byte 0       uint8 scale code m
bytes 1..47  376 sign bits, coefficient order, LSB-first
```

The scale is fully specified before source access:

```text
alpha = maxabs_f32(decoded_coarse_block) * m^2 / 2^18.
```

For `m>0`, sign bit 1 means `+alpha` and 0 means `-alpha`; transformed
coefficients 376 through 4095 are exactly zero. `m=0` is canonical only with
an all-zero sign field. Thus the legal codebook has
`1 + 255*2^376` canonical records. The scale byte is charged: each record is
exactly 8 scale bits plus 376 sign bits, with zero free or unallocated bits.

The conditional transform is the already-audited source-independent
SplitMix64 ordinal-17, 12-stage dyadic transform. Every finite correction lies
in `B[:,0:376]`, a strict subset of the audited continuous
`B[:,0:384]` span. The independent decoder transforms every reconstructed
correction back and fails closed unless its coefficient tail is numerical
zero and every 48-byte record re-encodes literally.

## Encoder search and the label-flexibility boundary

The encoder does not preserve predetermined fine labels. For every block it
selects all legal fine decisions from the source residual: it exhausts all
256 scale labels and chooses all 376 signs to minimize exact source-domain SSE
under the orthogonal transform. Since every legal message costs exactly 384
bits, `D + lambda R` reduces to exact `D` minimization in this frozen
codebook.

The executable also evaluates the same-codebook forced-local construction:
`sign(y)` plus the frozen scale nearest to `mean(abs(y))`. The two records must
match exactly. This is not a weakness of the search: for a fixed-length
hypercube with one shared amplitude, that local rule is the analytic global
nearest codeword. A mismatch fails closed.

V3 deliberately does **not** reoptimize the v6 coarse codeword, change coarse
labels, add variable-rate entropy decisions, or become a LOGIC-Q experiment.
Those require a new source freeze and a new dominance argument.

## Mandatory continuous hard gate

Finite encoding is dominated by its continuous parent. At literal 2.5 bpw,
with independently decoded coarse relative MSE `D0`, the exact required error
capture is

```text
c_required = max(0, 1 - 0.025/D0).
```

For the externally reported raw one-expert v6 coarse score
`D0=0.036975150060595235`, this evaluates to
`0.32387022205373717`, i.e. **32.3870222% of coarse SSE**. V3 always
recomputes the value from the independently decoded bytes; it does not reuse
the older 19.10% requirement from the different 2.5-bpw baseline. Even an
ideal Gaussian successive-refinement use of the entire
`2.5 - 2.3984375 = 0.1015625 bpw` would give only
`D=0.032119089632297364`, `F=1.0278108682335156`; hierarchy alone cannot
close this source-specific gap.

If the measured continuous `B[:,0:384]` capture is below this threshold, v3
publishes a completed hard-kill receipt and emits no composite. If rank 384
survives but active `B[:,0:376]` does not, it likewise stops. This kills only
the frozen DH384 codebook, not graph lifting, posterior reconstruction,
syndrome coding, joint coarse/fine search, or broader TACTIC-CAGE.

The containment diagnostics are reported separately:

- 384 dimensions are only `384/4096 = 9.375%` of an isotropic residual;
- a Gaussian one-bit/dimension sign reconstruction captures
  `(2/pi)*(384/4096) ~= 5.9683%` of total isotropic error;
- the implemented 376-dimensional sign subset has corresponding fractions
  `9.1796875%` and about `5.844%`.

These are diagnostics, not Qwen measurements or universal converses.

## Literal single-expert physical packet

V3 does not infer the earlier six-expert common packet:

```text
charged self-describing expert header      4,608 bytes =   1/128 bpw
exact v6 coarse frame                   1,414,656 bytes = 307/128 bpw
1,152 literal 48-byte fine records         55,296 bytes =  12/128 bpw
------------------------------------------------------------------------
literal one-expert composite             1,474,560 bytes = 320/128 bpw
```

The composite is exactly 360 aligned 4-KiB pages. One external pass over this
literal packet is 1x relative to its own physical bytes. No six-expert global
packet is emitted or parsed, and this package makes **no `73/72` claim**. It
also does not claim measured accelerator inference-HBM traffic; HBM remains a
separate audit.

The 4,608-byte header is real, parsed, canonical, and included in the
composite. It binds the coarse frame, fine stream, input manifest, completed
v6 result, and exact finite source closure. There is no metadata slack.

## Exact v6 dependency and launch authority

`V6_LOCK.json` pins the complete v6 source manifest/root, all 13 source
members, predecessor/runtime locks, and the exact `decode_tile_v6` source.
V3 retains no-follow descriptors for both source packages and the completed
v6 result throughout the run. The original BF16 Gate/Up/DownT files must match
the v6 `INPUT_BINDING.json` byte-for-byte.

The public authorization string is not authority. Before any v6 result or
BF16 file is opened, the dispatcher requires an externally hashed review
receipt with this canonical object shape:

```json
{
  "schema": "tactic-dh384-finite-v3-launch-review-v1",
  "status": "AUTHORIZE_ONE_BOUND_QWEN_GEOMETRY_EXPERT_FINITE_PILOT",
  "package_manifest_sha256": "<exact frozen v3 manifest>",
  "package_source_root_sha256": "<exact frozen v3 root>",
  "v6_source_manifest_sha256": "31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d",
  "v6_source_root_sha256": "161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2",
  "v6_complete_sha256": "<independently audited v6 COMPLETE.json>",
  "input_manifest_sha256": "<same input manifest bound by v6>",
  "allowed_scope": {
    "experts": 1,
    "geometry": [768, 2048],
    "qwen_or_model_identity_available_to_codec": false,
    "universal_tail_claim": false
  },
  "independent_audit": {
    "finite_source_reviewed": true,
    "v6_completed_result_reviewed": true,
    "payload_launch_explicitly_authorized": true
  },
  "review_claim_sha256": "sha256(canonical JSON of all preceding fields)"
}
```

The reviewer, not this package, creates that receipt after source and v6
result audits. Until then, do not launch the payload path.

## Source-only verification

After the manifest is frozen, these commands use only the standard library
and source files:

```bash
python -I -B research/tactic_dh384_finite_v3/verify_source.py
python -I -B research/tactic_dh384_finite_v3/test_source_only.py
```

They do not initialize CuPy or open a model/result payload. The production
dispatcher is CuPy-heavy only after retained source/runtime and external launch
review authentication.

## Reviewed pilot command

Only after an external reviewer issues the exact receipt:

```bash
CUDA_VISIBLE_DEVICES=0 python -I -B \
  research/tactic_dh384_finite_v3/dispatcher.py \
  --authorization RUN_REVIEWED_TACTIC_DH384_FINITE_V3_ONE_BOUND_EXPERT \
  --package-manifest-sha256 <v3 manifest sha256> \
  --repo-root /workspace/INT2_Q_C \
  --v6-package-dir /workspace/INT2_Q_C/research/tactic_actual_coarse_n18_v6 \
  --v6-result-dir <audited completed v6 result> \
  --input-manifest <exact v6 input manifest> \
  --launch-review <independent launch review> \
  --launch-review-sha256 <literal review file sha256> \
  --output-dir <absent output directory>
```

Publication uses a private staging directory, rehashes all members, performs
Linux `renameat2(RENAME_NOREPLACE)`, reopens and rehashes the public directory,
and only then renames a pending marker to terminal `COMPLETE.json`. A namespace
without terminal completion is not a result.

## Claim boundary

This is one Qwen-geometry expert pilot implementation. It does not establish
universal SwiGLU-MoE tails, non-Qwen portability, six-expert amortized traffic,
below-2x accelerator HBM, or `F <= 0.8`. Those become claims only if a literal
completed composite survives independent result audit.
