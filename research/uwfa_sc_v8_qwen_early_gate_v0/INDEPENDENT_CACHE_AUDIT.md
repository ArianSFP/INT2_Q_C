# Independent cache audit — 2026-09-02

Verdict: `PASS_FOR_NONPROMOTING_EARLY_KILL`.

Audited runner SHA-256:

```text
73e13572f8b3d3f852052ac54c7547239166e4b6a1209b6c11d884c0f08f8e9a
```

The single-artifact proxy removes only the duplicate
`StrataSCAdapter.extract_from_current` call.  The authenticated synchronous
call graph contains exactly two wrapper extractions: the outer geometry
binding and sealed `source_phase`.  The second call must have the identical
artifact byte count and SHA-256 and returns the exact first validated panel.
The result receipt requires exactly two wrapper calls and exactly one delegate
decode.

Independent source-free adversarial replay through sealed `prepare_panel`
confirmed equality of:

- selected bits, levels and bases;
- full and structural panel geometry;
- deterministic semantic-owner decoration;
- cached and fresh-decode panel values.

The runner is synchronous, the delegate has no callback to the proxy, and no
panel mutation occurs between the first preparation and `source_phase`.
Sealed `source_phase` recomputes both geometry hashes before any fit.  Final
decode also authenticates the regenerated source, reconstruction, canonical
stream re-encode and canonical container rebuild.

The cache is deliberately not claimed as a general production adapter.  It is
mutable and not thread-safe; a caught first-decode exception and arbitrary
third-party panel mutation are outside this exact uncaught call graph.  Those
generic limitations do not change a negative result from this explicitly
nonpromoting Qwen early-kill experiment.  A survivor still requires the
external production dispatcher, matched controls and independent result audit.

RunPod source-only result: 12/12 tests passed under isolated CPython.  Detached
launcher SHA-256:

```text
819615c6ba9c3e72f043414cc82187ca577bc3ebd9e5670e52da52e82575f9cf
```
