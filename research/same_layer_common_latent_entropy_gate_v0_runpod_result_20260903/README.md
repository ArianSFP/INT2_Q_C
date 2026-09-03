# Qwen same-layer common-latent result v0

This directory records the sole payload invocation authorized by the sealed
deployment review for `same_layer_common_latent_entropy_gate_v0`.

## Frozen evaluation

- Model: `Qwen/Qwen3-30B-A3B`, revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Panel: layer 15 experts `0,8,...,120`; Up and `Down.T`; 32 authenticated
  BF16 files and 50,331,648 scored weights.
- Device: NVIDIA GeForce RTX 5090 through CuPy 14.2.0.
- Result SHA-256:
  `21642374e5de79dc8014aeb6bda751d16eacbe3afcc1365b039e97231df7f1f0`.
- Status: `HARD_KILL_FAVORABLE_IDEAL_BELOW_TARGET`.

The deployment authority was for exactly one invocation and is consumed. The
deployment must not be run again under the existing review.

## Result

The best deliberately favorable conditional-entropy number was the
quaternary modal common label:

```text
favorable gross private-label saving       0.04703678191314046 bpw (Up/Down)
required favorable threshold               0.22933495044437175 bpw (Up/Down)
fraction of required threshold              20.5100800475459%
shortfall                                  0.18229816853123129 bpw (Up/Down)
```

That favorable number gives away the common latent, its model, selectors,
framing, and finite-coder losses. Once the complete common/private two-part
description is charged, the modal latents increase rate:

| Variant | Favorable gross gain | Charged two-part gain |
|---|---:|---:|
| Binary, charged-MDL plane choice | `0.01735424071683293` | `-0.01413604560380873` bpw |
| Binary, favorable-oracle plane choice | `0.029135979098880327` | `-0.031644270768117266` bpw |
| Quaternary modal label | `0.04703678191314046` | `-0.044250346597094335` bpw |

Negative charged gain means the common stream plus conditional private streams
is larger than coding the same expert/role label marginals independently. The
fixed scale stream is identical on both sides and is charged consistently.

The full target is `0.15288996696` bpw over equally sized Gate/Up/Down weights.
Because this gate includes only Up and Down, its necessary favorable threshold
is multiplied by `3/2`, giving `0.22933495044437175` bpw.

## Routed-read projection

The read calculation uses 4,096-byte pages and charges the union of the global
common section and the selected expert's private section. The maximum is the
worse of physical-ownership and non-padding-decodable-ownership amplification.
These are page-layout projections, not an emitted packet or measured HBM
traffic.

| Variant | Requested rate | Actual page rate | Common pages | Maximum amplification | Read result |
|---|---:|---:|---:|---:|---|
| Binary charged-MDL | `2.15` | `2.150390625` | 98 | `1.5919152753403583x` | strict pass |
| Binary charged-MDL | `2.5` | `2.5` | 98 | `1.7676116100769195x` | strict pass |
| Quaternary | `2.15` | `2.150390625` | 194 | `2.0371413557809466x` | fail |
| Quaternary | `2.5` | `2.5` | 194 | `2.209957974651824x` | fail |

Thus the binary representation demonstrates that a same-layer common/private
layout can remain below `2x`, but it has no rate benefit on this panel. The
largest favorable source signal is quaternary and is both far below the
scientific threshold and above the strict routed-read cap.

## Decision boundary

The run correctly stopped before all eight affine-scramble controls and before
finite-coder construction. It is an ideal-label MDL aperture, not a codec and
not an MSE result.

This closes only fixed four-level, identity-coordinate, modal binary or
quaternary common latents on the frozen 16-expert Up/Down panel. It does not
close expert clustering, cycle-consistent neuron alignment, hierarchical
group-common streams, flexible joint label selection, Gate coupling, a learned
procedural latent, or lossy Gray-Wyner reconstruction.
