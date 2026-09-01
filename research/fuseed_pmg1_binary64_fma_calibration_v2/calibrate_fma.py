#!/usr/bin/env python3
"""Explicit-FMA binary64 successor to the frozen PMG1 stage-0 calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_TEMPLATE_SHA256 = "9376720ec812b93e070ccb93433e83ff243213d6d244c7a18afa84b3d8690c24"
COMPILE_OPTIONS_V2 = (
    "--std=c++17",
    "--fmad=true",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-I/usr/local/cuda/include",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_template_namespace():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "fuseed_pmg1_binary64_calibration_v1"
        / "calibrate_binary64.py"
    )
    actual = sha256_file(template_path)
    if actual != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError(f"binary64 template hash mismatch: {actual}")
    source = template_path.read_text(encoding="utf-8")
    replacements = (
        (
            '"fuseed_pmg1_binary64_source_free_stage0_calibration_v1"',
            '"fuseed_pmg1_binary64_explicit_fma_stage0_calibration_v2"',
            1,
        ),
        (
            '"BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT"',
            '"EXPLICIT_FMA_BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT"',
            1,
        ),
        (
            '"EARLY_KILL_BINARY64_STAGE0_NO_QWEN"',
            '"EARLY_KILL_EXPLICIT_FMA_BINARY64_STAGE0_NO_QWEN"',
            1,
        ),
        (
            '"fuseed_pmg1_binary64_shard_journal_v1"',
            '"fuseed_pmg1_binary64_explicit_fma_shard_journal_v2"',
            1,
        ),
        (
            '"fuseed_pmg1_binary64_stage0.cu"',
            '"fuseed_pmg1_binary64_explicit_fma_stage0.cu"',
            1,
        ),
        (
            '"fuseed_pmg1_binary64_parity.cu"',
            '"fuseed_pmg1_binary64_explicit_fma_parity.cu"',
            1,
        ),
    )
    for old, new, expected in replacements:
        count = source.count(old)
        if count != expected:
            raise RuntimeError(f"template label replacement cardinality mismatch: {old}: {count}")
        source = source.replace(old, new)
    namespace = {
        "__name__": "fuseed_pmg1_binary64_explicit_fma_impl",
        "__file__": str(Path(__file__).resolve()),
    }
    exec(compile(source, str(template_path), "exec"), namespace)
    namespace["template_python_sha256"] = actual
    namespace["derived_python_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return namespace


def explicit_fma_deriver(template_deriver):
    def derive(source: str):
        source, counts = template_deriver(source)
        counts = dict(counts)
        replacements = (
            (
                '''  const double centered_x2 =
      sum_x2[fit_cat] - sum_x[fit_cat] * sum_x[fit_cat] / (double)fit_n;
  const double centered_wx =
      sum_xw[fit_cat] - sum_x[fit_cat] * sw_fit / (double)fit_n;''',
                '''  const double centered_x2 = __dsub_rn(
      sum_x2[fit_cat], __ddiv_rn(
          __dmul_rn(sum_x[fit_cat], sum_x[fit_cat]), (double)fit_n));
  const double centered_wx = __dsub_rn(
      sum_xw[fit_cat], __ddiv_rn(
          __dmul_rn(sum_x[fit_cat], sw_fit), (double)fit_n));''',
                "centered_moments_rn",
                1,
            ),
            (
                '''  const double mu_raw = mean_w - alpha_raw * sum_x[fit_cat] / (double)fit_n;''',
                '''  const double mu_raw = __dsub_rn(
      mean_w, __ddiv_rn(
          __dmul_rn(alpha_raw, sum_x[fit_cat]), (double)fit_n));''',
                "mu_rn",
                1,
            ),
            (
                '''  double sse = sw2_score + (double)score_n * mu * mu
      + alpha * alpha * sum_x2[score_cat]
      + 2.0 * mu * alpha * sum_x[score_cat]
      - 2.0 * mu * sw_score - 2.0 * alpha * sum_xw[score_cat];''',
                '''  double sse = sw2_score;
  sse = __fma_rn((double)score_n, __dmul_rn(mu, mu), sse);
  sse = __fma_rn(__dmul_rn(alpha, alpha), sum_x2[score_cat], sse);
  sse = __fma_rn(
      __dmul_rn(2.0, __dmul_rn(mu, alpha)), sum_x[score_cat], sse);
  sse = __fma_rn(__dmul_rn(-2.0, mu), sw_score, sse);
  sse = __fma_rn(__dmul_rn(-2.0, alpha), sum_xw[score_cat], sse);''',
                "sse_explicit_fma",
                1,
            ),
            (
                '''  return sw2_score - 2.0 * mean_w * sw_score
      + (double)score_n * mean_w * mean_w;''',
                '''  double baseline = sw2_score;
  baseline = __fma_rn(__dmul_rn(-2.0, mean_w), sw_score, baseline);
  baseline = __fma_rn(
      (double)score_n, __dmul_rn(mean_w, mean_w), baseline);
  return baseline;''',
                "baseline_explicit_fma",
                1,
            ),
            (
                '''        local_sum_x2[category] += x * x;''',
                '''        local_sum_x2[category] = __fma_rn(
            x, x, local_sum_x2[category]);''',
                "anchor_square_explicit_fma",
                1,
            ),
            (
                '''            sum_xw0[category] += x * (double)__ldg(
                &targets[domain0 * VALUE_COUNT + coordinate]);''',
                '''            sum_xw0[category] = __fma_rn(
                x, (double)__ldg(
                    &targets[domain0 * VALUE_COUNT + coordinate]),
                sum_xw0[category]);''',
                "domain0_cross_explicit_fma",
                1,
            ),
            (
                '''            sum_xw1[category] += x * (double)__ldg(
                &targets[domain1 * VALUE_COUNT + coordinate]);''',
                '''            sum_xw1[category] = __fma_rn(
                x, (double)__ldg(
                    &targets[domain1 * VALUE_COUNT + coordinate]),
                sum_xw1[category]);''',
                "domain1_cross_explicit_fma",
                1,
            ),
            (
                '''  const double capture = 1.0 - (sse / baseline);''',
                '''  const double capture = __dsub_rn(1.0, __ddiv_rn(sse, baseline));''',
                "capture_rn",
                1,
            ),
        )
        for old, new, label, expected in replacements:
            count = source.count(old)
            counts[label] = count
            if count != expected:
                raise RuntimeError(f"explicit-FMA replacement cardinality mismatch: {label}: {count}")
            source = source.replace(old, new)
        forbidden = (
            "local_sum_x2[category] += x * x",
            "sum_xw0[category] += x *",
            "sum_xw1[category] += x *",
        )
        if any(value in source for value in forbidden):
            raise RuntimeError("implicit dominant multiply-add survived explicit-FMA derivation")
        counts["explicit_fma_source_occurrences"] = source.count("__fma_rn(")
        counts["rounding_intrinsic_source_occurrences"] = sum(
            source.count(name) for name in ("__dadd_rn(", "__dsub_rn(", "__dmul_rn(", "__ddiv_rn(")
        )
        return source, counts

    return derive


def main() -> None:
    namespace = load_template_namespace()
    template_deriver = namespace["derive_binary64_capture_source"]
    namespace["derive_binary64_capture_source"] = explicit_fma_deriver(template_deriver)
    namespace["COMPILE_OPTIONS"] = COMPILE_OPTIONS_V2
    namespace["main"]()


if __name__ == "__main__":
    main()
