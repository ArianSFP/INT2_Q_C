# PAIRPATH-P2 r3 source-only repair

Status: **SOURCE-ONLY HOLD — INDEPENDENT HOSTILE AUDIT REQUIRED.**

This package repairs the three blockers found by the independent audit of r2.
It opens no Qwen payload, has no GPU or network path, and grants neither a
payload capability nor hard-kill authority.

## Repair 1: global finite objective

For a two-expert source `x[e,role,i]`, the finite Up and Down fitting passes use
one multiplier

```
w = lambda * sum(x[:,(up,down)]**2) / x[:,(up,down)].size
```

The multiplier is computed once in `_make_plan`, passed explicitly to both role
fits, and serialized in `r3_encoder_certificate`. The decoder rejects a packet
whose two optimized-role certificates differ. A deliberately unequal-role-
energy KAT instruments every fold fit and proves it receives the exact same
IEEE-754 value.

## Repair 2: dominance, without a false converse

For every lambda and role, the independent flexible-label solution is inserted
verbatim into the joint solver's candidate set. Its distortion and empirical
joint entropy are rescored under the exact same joint objective. The returned
joint solution is accepted only if

```
J_joint(returned) <= J_joint(independent labels) + roundoff tolerance.
```

Every row carries that certificate. This proves candidate dominance, not global
optimality. The alternating entropy-constrained search is still heuristic, so a
small measured gain now reports `HEURISTIC_BELOW_GATE_NONAUTHORITATIVE`; r3 can
never issue a hard kill from this oracle.

## Repair 3: causal tree replay

Before any literal packet is decoded, `_parse_packet` now:

1. decodes the packed matching and merge ranks;
2. materializes the pair-first binary tree;
3. canonically re-encodes it;
4. requires exact agreement with `packed`, `bits`, `pairs`, `merge_ranks`, and
   `materialized`.

Malformed, inconsistent, unused, noninteger, or unreplayed descriptions fail
closed. The KAT mutates every redundant tree field inside a freshly canonical,
page-valid packet and verifies rejection.

## Pinned dependency

`pairpath_r3_core.py` executes the exact sealed r2 source as a base and refuses
to import it unless its SHA-256 is

`2c99a31aef669cabbb67137061233640b013e8c50a5132ddbcc9ffec2c239034`.

The r3 manifest separately binds this dependency. This keeps r2 and its audit
immutable while making the small repair delta inspectable.

## Tests

Run only the deterministic source-free suite:

```
C:\INT2__compression\.venv-cupy\Scripts\python.exe -I -B research\pairpath_fl_same_layer_microcodec_v0_20260904_r3\test_source_only.py
```

The sealed receipt records nine passing tests. No result in this package is
evidence about Qwen or any other model.
