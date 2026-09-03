# INT2_Q / VORPAL repository audit

Date: 2026-09-03

## Disposition

The external `ArianSFP/INT2_Q` repository has been inspected rather than
treated as an unevaluated source of ideas.  Its current `main` is commit
`84cb19eb590d081ca5d124a5a586c97b16a56836`, also tagged
`vorpal-role-joint-negative-v1`.  The earlier base endpoint is tag
`vorpal-negative-gap-v1` at commit
`481d02a45bbc4cf293f9bac88c0be7bbe6135eaf`.

The published standard-library verifiers were executed locally and passed for
both the 400-block VORPAL bundle and its role-joint wrapper.  These checks
authenticate the physical objects and their bound exact-source receipts; they
do not independently re-open the omitted Qwen BF16 payloads.

VORPAL/SPARC is **not** a primary route to the present universal SwiGLU-MoE
target.  Its reusable pieces are finite-code engineering modifiers, not the
missing source advantage needed for `F <= 0.8`.

## Exact published endpoints

The base VORPAL development artifact reports:

```text
values             104,857,600
physical bytes      32,583,835
R                    2.4859493255615234 bpw
D                    0.02983268081273826
F = D * 2^(2R)       0.936230770932366
gap                  -0.2861708909351923 dB
```

The role-joint wrapper reports:

```text
physical bytes      32,767,988
R                    2.4999990844726563 bpw
D                    0.02935406047291881
F                    0.939328742945628
gap                  -0.2718238828655344 dB
```

Both are well above the target `F <= 0.8`, whose signed gap is approximately
`-0.9691 dB`.  They are post-hoc results on a 400-block development panel, not
full-checkpoint or disjoint confirmation evidence.

The favorable base aggregate is not a SwiGLU-expert result.  Recomputing the
published role totals at the common base rate gives approximately:

| Published role | `F` |
|---|---:|
| Expert Down | `0.99865` |
| Expert Up | `1.02970` |
| Expert Gate | `1.05208` |
| Attention V | `0.69798` |
| Three expert roles pooled | approximately `1.0` |

Thus much of the aggregate negative gap comes from non-expert roles, especially
attention V.  The role-joint extension brings each sampled expert-role
aggregate barely below the Gaussian reference but leaves their pooled result
approximately neutral (`F ~= 0.99962`).

## Procedural SPARC stage: rate-distortion audit

The extension sends one 242-byte stage as 96 twenty-bit signed-Hadamard atom
addresses plus one FP16 amplitude.  Its fixed wrapper/coordinate portion is
1,201 bytes.  The three exact coordinate gains sum to
`0.3422223388072041` SSE.  Applying those coordinate corrections without any
SPARC stage changes normalized performance from:

```text
base F              0.936230770932366
coordinate-only F   0.9361854209175292
```

This is a real but tiny positive modifier.

At that panel size, another 242 bytes must remove approximately
`0.0499155` SSE merely to keep `F` flat.  The emitted FP16 amplitudes imply
first-stage gains close to `96*a^2`:

| Role | First-stage SSE gain | Largest gain in emitted prefix |
|---|---:|---:|
| Up | `0.0423818` | `0.0431237` |
| Down | `0.0428138` | `0.0432480` |
| Gate | `0.0414026` | `0.0427519` |

The sum of the amplitude-derived gains agrees with each published exact role
total to a few parts in `10^5`.  Every emitted stage is below break-even.
Optimizing the complete published prefix bank for the global `F` objective
therefore selects **zero stages**.  The frozen 756-stage role-balancing wrapper
worsens global `F` to `0.939328742945628`.

This does not prove that every future procedural codebook is useless.  It does
close this frozen SPARC4 stage family as a standalone transplant for the
present rate-relative objective.

## Routed-read incompatibility

VORPAL stably sorts 2,048-value groups by reconstructed variance across the
entire 400-block panel and forms causal arithmetic-coded chunks that cross
tensor, semantic-role, and layer boundaries.  The published result therefore
cannot be assigned an expert-local read ratio.

Using the published group-to-chunk manifest and literal container sizes, a
favorable lower bound for reconstructing one 262,144-value source block is:

| Population | Chunks touched | Lower-bound read amplification |
|---|---:|---:|
| All 400 blocks | `4..35`, mean `14.57` | `3.92x..32.03x`, mean `13.98x` |
| 144 expert-role blocks | `4..22`, mean `9.16` | `3.92x..18.16x`, mean `9.04x` |

The bound excludes common headers and the role extension, so it is favorable.
Every sampled expert-role block already exceeds the strict `<2x` contract.
An expert-local re-pooling experiment would be a different codec and cannot
inherit the published distortion result.

## Overlap with the current repository

Most VORPAL mechanisms already have local counterparts or stronger negative
screens:

- global energy ranking and long polar pooling: STRATA/POLARIS;
- reverse waterfilling and physical profile selection: the existing profile
  and rate-allocation searches;
- scale/log-variance side models: conditional-hyperprior and spectral-scale
  experiments;
- sparse exact tails: the tail-peeling experiments;
- Gate/Up/Down joint structure: role KLT and invariant-manifold experiments;
- procedural signed bases: procedural-subspace and FUSEED experiments;
- overflow reservoirs: the current fixed packet/reservoir formats;
- expert-local physical accounting: the current approximately `1.169x`
  layout, which is materially better for MoE reads.

## Retained bounded candidates

1. **A64-to-A128 chunk upgrades.**  This is the clearest exact VORPAL
   mechanism not yet identified byte-for-byte in the local finite codec.  It
   merits a small overload-heavy aperture, with an early kill tied to actual
   `F` improvement and expert-local reads.
2. **Fixed-route integer-byte Pareto selection.**  Reuse as implementation
   machinery after a source model survives; it is not itself a source gain.
3. **Rice-coded coordinate pulses.**  Retain only as last-mile slack filling;
   the published positive effect is extremely small.
4. **Fine relative log-variance labels.**  At most a bounded expert-local
   ablation because the local free-field oracles already limit the likely gain.

## Conclusion

`INT2_Q` contains a real, verifiable finite compression result and good audit
engineering.  It does not contain an untested path that can simply be imported
to meet the current universal SwiGLU-MoE objective.  VORPAL's expert source
advantage is weak, its global pooling violates routed locality, and the frozen
SPARC refinement is rate-relative adverse.  The main research priority remains
same-layer clustered/flexible-label coding and a true source model that can
deliver at least the missing `0.11356` to `0.15289` bpw-equivalent advantage
inside one expert-local physical ledger.
