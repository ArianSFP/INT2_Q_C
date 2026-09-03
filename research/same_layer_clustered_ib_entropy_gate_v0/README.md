# Same-layer clustered information-bottleneck entropy gate v0

This directory is a sealed, source-only design for **CBIB-1**: a strict test of
whether experts in the same SwiGLU-MoE layer contain useful conditional entropy
that a routed expert can consume below `2x` read amplification.

It is not a codec, a Qwen result, a deployment package, or execution authority.
`run_gate.py` contains no payload locator and has
`PAYLOAD_EXECUTION_ENABLED = False`.  There is no enabled payload branch.

## Frozen scientific aperture

- Labels have shape `[expert, 2, coordinate]`: canonical Up and `Down.T` roles,
  with a fixed four-symbol alphabet.
- Flat group sizes are exactly `{2,4,8,16}`; only sizes dividing the evaluated
  expert count are legal.
- Every coordinate is assigned by `floor(coordinate/2048) mod 8`, cycling over
  all 2,048-value superblocks in the tensor.  Thus there are eight folds, not
  only eight blocks.  For fold `f`, the partition, latent model, marginal model,
  and all model choices use only the other seven folds.  Fold `f` is scored once
  and is never used to select its own model.
- Expert clustering uses a deterministic greedy equal partition built from
  training-only pairwise four-symbol mutual information.  Every chosen
  partition is transmitted and charged by the exact equal-partition count

  `E! / ((k!)^(E/k) (E/k)!)`.

- Within each group and role, hard EM fits a two-state product model.  Every
  coordinate may choose either binary state; the state is not constrained to a
  modal symbol or Gray bitplane.  Frozen starts include every expert Gray plane,
  modal planes, parity planes, and a public hash split.  This is an arbitrary
  learned binary latent within the frozen finite search, not a proof of the
  globally optimal binary latent.
- Probabilities use a fixed Krichevsky-Trofimov half-count.  One latent count and
  three of four categorical counts are transmitted; omitted counts are exactly
  derived.  Lower state wins every assignment tie.

The favorable source gate is

`(independent marginal data bits - private conditional data bits) / UpDown weights`.

It grants the binary common stream, models, partition, selectors, headers,
framing, pages, and coder loss for free.  If it is below the exact
**0.22933495044437175 Up/Down bpw** threshold, the branch hard-kills before any
control.  No lower triage line can promote.

If the favorable gate and one physical/read endpoint survive, eight fixed
per-expert/per-role affine coordinate permutations rerun the complete
partition/model search.  These controls exactly preserve each marginal label
histogram.  Promotion requires the fully charged, maximum-control-corrected
gain to meet the same exact threshold.  A survivor still requires one literal
finite codec and original-domain MSE reconstruction; this source census cannot
establish `F <= 0.8` by itself.

## Exact charges

The independent description charges held-out marginal NLL, three-count model
descriptors, and one 256-byte private frame per expert.  CBIB-1 charges:

- held-out latent and private conditional NLL;
- every latent and conditional count descriptor;
- eight enumerative partition descriptors;
- the exact selector for the compatible frozen group-size bank;
- one 4,096-byte global frame;
- one 256-byte frame for every fold/group common segment;
- one 256-byte private frame per expert.

Scale bytes are identical and therefore cancel from the MDL gain, but they are
explicitly included in every private-stream capacity and read calculation.

## Physical and non-padding read ledgers

For expert `e`, let `V(e)` be its one flat group segment in each fold.  With
page-rounded bytes `Gbar`, `Cbar_v`, `Pbar_e`, unpadded decodable bytes
`G`, `C_v`, `P_e`, and group size `k_v`, the exact ledgers are

```
T_e       = Gbar + Pbar_e + sum_{v in V(e)} Cbar_v
D_phys,e  = Gbar/E + Pbar_e + sum_{v in V(e)} Cbar_v/k_v
D_np,e    = G/E    + P_e    + sum_{v in V(e)} C_v/k_v
A_phys,e  = T_e / D_phys,e
A_np,e    = T_e / D_np,e
```

Both amplifications must be **strictly** below `2`.  The numerator always uses
the union of touched pages, so padding cannot hide traffic.  Exact page budgets
are checked at `43/20` and `5/2` bpw.  An envelope is only an ideal capacity
projection until a finite packet and instrumented decoder exist.

The design is universal across SwiGLU MoEs: it uses only expert count, matrix
shape, role, coordinates, and transmitted source-model fields.  It uses no base
checkpoint, fine-tuning ancestry, external expert, model identity, or Qwen-only
decoder state.  The eventual Qwen panel is an evaluation target, not part of
this source-only package.

## Verification

The external caller must pin the manifest digest:

```powershell
$pkg = "C:\INT2__compression\INT2_Q_C\research\same_layer_clustered_ib_entropy_gate_v0"
$manifest = (Get-FileHash -Algorithm SHA256 -LiteralPath "$pkg\SOURCE_MANIFEST.json").Hash.ToLowerInvariant()
python -B "$pkg\verify_source.py" --package "$pkg" --manifest-sha256 $manifest
python -B "$pkg\test_source_only.py"
```

The optional mandatory-before-deployment GPU preflight has no payload argument:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
& "C:\INT2__compression\.venv-cupy\Scripts\python.exe" -B "$pkg\run_source_free_cupy.py" `
  --fixture-token RUN_SOURCE_FREE_CBIB1_FIXTURE_V0 `
  --source-manifest-sha256 $manifest
```

Any future payload runner must be implemented as a separately named package,
manifest-pinned, independently reviewed, and separately authorized.  Copying or
editing this source package is not execution authority.
