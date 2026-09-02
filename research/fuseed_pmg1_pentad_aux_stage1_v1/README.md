# Fixed PMG1 pentad auxiliary stage-1 source v1

Status: **frozen source-only; awaiting independent audit; no payload or run
authority.**

This is a fresh successor package for one cheap, decisive test of the five
explicit PMG1 seeds below. It does not modify or supersede the historical
pentad or tetrad packages.

```text
3306464084, 235286348, 2174751347, 256779041, 118211936
```

The test does **not** rerun or authenticate the historical `2^32` heuristic
selection. It treats those five integers as public constants of one fixed
hypothesis. This is a legitimate narrow test because neither the tuple nor the
frozen 2,048 fit and 2,048 score coordinates can change after seeing the new
scores. A failure kills only this tuple; a survivor proves only that this tuple
deserves separately authorized stage-2 review.

## Exact experiment

The source-only `plan_snapshot.py` reconstructs the historical auxiliary
coordinate plan and must reproduce:

| split | count | SHA-256 of canonical newline-delimited keys |
|---|---:|---|
| fit | 2,048 | `42a2cb8170a1de43f23ee399ef93687cf738fce42d9f314b9273839855961f9e` |
| score | 2,048 | `c112da528fcedfbcf62a0f71ea3f63150cc2b68a0bc7b0d34d63290580f0d7bb` |

Those coordinates cover 23 layer-15 Qwen development identities: Up and Down
for experts 0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, and 112, except that
expert 0/Up is absent from the historical plan. Gate is not opened.

For each identity independently, the producer must regenerate all five
anchors through the exact CuPy/NVRTC PMG ABI, fit one centered 5×5 ridge
system on that identity's fit coordinates, round all five coefficients and
the intercept to IEEE FP16, decode those literal words, and score one joint
reconstruction on only its disjoint score coordinates. There is no summing of
the five marginal captures.

The uncertainty unit is one whole expert, not one matrix. The gate deletes
all roles belonging to each of the 12 experts in turn. Sixteen frozen NumPy
PCG64 controls independently permute anchor rows within every identity and
split and refit the full model.

A survivor requires all of the following:

- pooled raw capture minus three delete-expert standard errors is at least
  `0.1910966610577134`;
- all 23 individual identity captures and both pooled role captures are
  strictly positive;
- the primary aggregate capture exceeds every frozen scramble control;
- all numerical and reconstruction checks pass.

The strict status is otherwise
`HARD_KILL_FIXED_PENTAD_AUXILIARY_STAGE1_NO_TUPLE_RETRY`.

## Why this is low-priority but worth closing

The independently audited four-seed predecessor scored a Qwen capture of
`-0.04577526835279766`; its historical delete-one-**matrix** upper three-SE
bound was only `0.0015538337327504342`. That is strongly adverse. The fifth
seed would need a spectacular joint reversal to reach `0.1910966610577134`.
This gate is nevertheless cheap: only `23 × 4,096 × 5 = 471,040` procedural
anchor coordinates are generated for the primary, and the controls reuse
them. It closes the exact pentad without spending an unauthenticated `2^32`
search again.

## Physical planning boundary

If a later full codec existed, the frozen six-expert metadata ledger would be
320 bytes (`80 + 6×40`), `0.0000904224537037037` bpw. Debiting those bits from
the residual payload keeps the planned total at 2.5 bpw. Charging two new
4-KiB pages per route gives a conservative `1.175×` page-read amplification.
No pentad container exists, so none of these planning numbers is a measured
codec result.

## Inert by construction

This package contains pure contracts, plan reconstruction, a CuPy anchor
primitive, an in-memory evaluator, tests, and a source verifier. It contains
no source loader, payload path, SSH or network client, output writer, or
payload-running CLI. Running `inert_entrypoint.py` only emits the refusal
receipt and exits nonzero.

Before any auxiliary payload is touched, all six launch blockers in
`design_lock.json` must be resolved by work owned outside this package. In
particular, a separate independent audit and a separate explicit scope-limited
authorization are mandatory.

## Source-only verification

After `SOURCE_MANIFEST.json` is frozen, run locally or in an empty source-only
environment:

```bash
python3 -B test_source.py
python3 -B verify_source.py
python3 -B inert_entrypoint.py
```

The last command must exit with status 3. None of these commands opens Qwen
weights or authorizes a future launch.

## Claim boundary

This package is not a Qwen result, Gate evidence, fresh validation, a family
screen, a residual-codec score, a bitstream, or evidence that `F <= 0.8` was
achieved. The five historical marginal captures totaling about 0.237 are not
a joint result. A future auxiliary survivor would still have exactly the same
narrow non-evidence boundary.
