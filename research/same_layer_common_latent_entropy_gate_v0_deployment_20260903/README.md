# Same-layer common-latent entropy gate v0

This is a source-only, fail-closed aperture for one previously untested question:
do fixed four-level Up and `Down.T` labels at identical coordinates carry enough
conditional entropy across 16 experts in Qwen layer 15 to justify a finite
common/private expert packet?

It is **not** a codec, a target result, or payload evidence. The checked-in
entrypoint has `PAYLOAD_EXECUTION_ENABLED = False`. Editing that literal changes
the source hash and is never execution authority. An authorized run must use a
separately copied, manifest-pinned deployment sibling and independent review.

## Frozen scope

- Model: `Qwen/Qwen3-30B-A3B`, revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Layer 15 experts `0,8,...,120`; exactly Up and `Down.T`.
- Every one of the 32 input files has an explicit relative path, byte count,
  raw/canonical shape, and SHA-256 in `panel_lock.json`. There is no discovery.
- Each canonical role is row-major and divided into 2,048-value blocks. FP64
  RMS is rounded to a transmitted IEEE binary16 scale. The decoded scale drives
  both fixed four-level thresholds and reconstruction values.
- The universal scoring core accepts any legal positive SwiGLU dimensions and
  `E` from 2 through 256. The Qwen panel is only an evaluation plugin.

The two common-latent families are deliberately small and decoder-explicit:

1. Binary: modal Gray-code bitplane at each coordinate, separately for Up and
   Down, with lower-symbol ties. Both Gray planes are exhausted.
2. Quaternary: modal four-level label at each coordinate, with lower-symbol
   ties.

Coordinates remain in identity alignment. Failure closes only this panel,
quantizer, alignment, and modal latent family. It does not close Gate, arbitrary
latent clustering, expert permutations, other layers, nonlinear common models,
or activation-aware objectives.

## Measurements and decision order

For the identical marginal labels, the aperture reports:

- independent per-expert/per-role plug-in ideal entropy;
- conditional ideal entropy given `U`;
- entropy of `U`;
- exact fixed-width two-part count descriptors (the final multinomial count is
  derived);
- identical charged scale bits;
- binary plane and binary/quaternary family selector bits;
- canonical integer count tables sufficient for independent recomputation.

The first number is intentionally favorable:

`(marginal ideal label bits - private conditional ideal bits) / UpDown weights`.

It grants `U`, models, selectors, framing, and coder losses for free. Therefore,
if both families fall below **0.22933495044437175 bpw on Up/Down**, they cannot
close the whole-SwiGLU target through this aperture and the run hard-kills before
controls or finite-coder work. The **0.045 bpw** line is triage information only;
it never promotes. Binary favorable-plane selection and charged-MDL plane
selection are reported separately because their optima can differ.

Only a source result clearing the full threshold runs eight fixed, independent
per-expert/per-role affine coordinate scrambles. Each scramble is bijective and
exactly preserves every marginal label histogram.

## Routed-read envelope

The prospective page layout is:

`[one global header + raw U + common model][one private stream per expert]`.

For expert `i`, exact touched bytes are the union of the page-aligned common
section and private section `i`. The report gates on the worse of:

- touched bytes divided by page-amortized physical ownership;
- touched bytes divided by non-padding decodable ownership.

Both must be strictly below `2x`. This prevents unused page fill from producing
a false pass. A family is eligible only if at least one of the two frozen rate
endpoints (`2.15` or `2.5`) reports adequate private capacity and passes both
strict read tests. Only those read-eligible families enter the charged-MDL and
control promotion decision; controls are skipped when no source family can
already pass the scientific, capacity, and read gates. Rates are exact rational
page envelopes in `[2.15,2.5]` bpw. They remain projections until a finite coder
emits a container and an instrumented decoder confirms its reads.

For the frozen shape there are 50,331,648 scored Up/Down weights: 3,303 pages at
the 2.15 floor (actual `1101/512 = 2.150390625` bpw) and 3,840 pages at 2.5.
The binary and quaternary common sections occupy 98 and 194 pages after their
model/selector bits.

## Source verification and source-free CuPy test

Use the externally recorded manifest digest; it is intentionally mandatory:

```powershell
$pkg = "C:\INT2__compression\INT2_Q_C\research\same_layer_common_latent_entropy_gate_v0"
$manifest = (Get-FileHash -Algorithm SHA256 -LiteralPath "$pkg\SOURCE_MANIFEST.json").Hash.ToLowerInvariant()
& "C:\INT2__compression\.venv-cupy\Scripts\python.exe" -B "$pkg\verify_source.py" --package "$pkg" --manifest-sha256 $manifest
$env:CUDA_VISIBLE_DEVICES = "0"
& "C:\INT2__compression\.venv-cupy\Scripts\python.exe" -B "$pkg\run_source_free_cupy.py" --fixture-token RUN_SOURCE_FREE_COMMON_LATENT_FIXTURE_V0 --source-manifest-sha256 $manifest
```

The fixture runner has no payload-root option. It checks CPU/CuPy scale bits,
labels, integer counts, both binary objectives, and two-part bits.

The initial frozen snapshot was independently blocked because it calculated
these read envelopes without consulting them in its promotion status. The
repaired source adds an explicit fail-closed family/rate selector and a
mandatory regression proving that four failed envelopes cannot return a
survivor status. The earlier negative audit remains immutable evidence about
that superseded snapshot.

## Future authorized payload command (do not run from this source release)

After copying to a separately named deployment directory, flipping the HOLD
only there, regenerating/pinning its manifest, and obtaining independent review:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -B /absolute/deployment/run_gate.py \
  --authorization EXECUTE_AUTHENTICATED_QWEN_L15_COMMON_LATENT_V0 \
  --payload-root /absolute/pinned/payload/root \
  --output /absolute/empty/output/common_latent_result.json
```

Authentication and BF16 decoding share one in-memory byte buffer, so each panel
file receives one application-level host scan. That is not claimed to measure
filesystem-page or HBM traffic. If the favorable threshold misses, the result
contains no scramble controls and no finite-coder construction.
