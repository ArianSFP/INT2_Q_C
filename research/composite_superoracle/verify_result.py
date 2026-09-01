#!/usr/bin/env python3
"""Independent verifier for the nested Qwen composite super-oracle.

The verifier does not import the experiment.  It rehashes and reloads all 18
BF16 sources, independently rebuilds every role/STRATA geometry, recomputes all
324 singular spectra and polar rank curves, and repeats every selected joint
reverse-waterfill and source-leaky envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
ROLES = 3
EXPERTS = 6
MATRICES = 18
STRATA = 8
ROWS_PER_STRATUM = ROWS // STRATA
VALUES_PER_MATRIX = ROWS * COLS
PANEL_VALUES = MATRICES * VALUES_PER_MATRIX
TARGET_F = 0.8
GLOBAL_HEADER_BITS = 4096 * 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def close(a: float, b: float, *, rtol: float = 3e-10, atol: float = 3e-10) -> None:
    if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
        raise AssertionError((a, b))


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.all(np.isfinite(values)):
        raise AssertionError(path)
    return values.reshape(shape)


def load_sources(root: Path, result: dict[str, Any]) -> tuple[Path, dict[str, Any], list[np.ndarray]]:
    lock_path = root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    if sha256_file(lock_path) != result["audit"]["source_lock_file_sha256"]:
        raise AssertionError("source lock file hash")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    clean_lock = dict(lock)
    claimed_lock = clean_lock.pop("lock_sha256")
    if hashlib.sha256(canonical(clean_lock)).hexdigest() != claimed_lock:
        raise AssertionError("source lock internal seal")
    if claimed_lock != result["audit"]["source_lock_internal_sha256"]:
        raise AssertionError("source lock identity")
    receipts = result["audit"]["source_receipts"]
    if len(receipts) != MATRICES or len(lock["matrices"]) != MATRICES:
        raise AssertionError("panel size")
    matrices: list[np.ndarray] = []
    for ordinal, (row, receipt) in enumerate(zip(lock["matrices"], receipts, strict=True)):
        if int(row["matrix_ordinal"]) != ordinal or int(receipt["matrix_ordinal"]) != ordinal:
            raise AssertionError("ordinal")
        path = lock_path.parent / row["output_relpath"]
        observed = sha256_file(path)
        if observed != row["source_bf16_sha256"] or observed != receipt["observed_sha256"]:
            raise AssertionError((ordinal, "source hash"))
        matrix = read_bf16(path, tuple(int(x) for x in row["shape"]))
        if row["role"] == "down":
            matrix = matrix.T
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        if matrix.shape != (ROWS, COLS):
            raise AssertionError(matrix.shape)
        matrices.append(matrix)
    return lock_path, lock, matrices


def role_transform(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = source.reshape(ROLES, -1)
    eigenvalues, basis = np.linalg.eigh(flat @ flat.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]
    for column in range(ROLES):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return (basis.T @ flat).reshape(source.shape), basis, eigenvalues


def strata(matrix: np.ndarray) -> list[np.ndarray]:
    energy = np.sum(np.square(matrix), axis=1, dtype=np.float64)
    order = np.lexsort((np.arange(ROWS, dtype=np.int64), energy))
    return [
        np.ascontiguousarray(order[s * ROWS_PER_STRATUM : (s + 1) * ROWS_PER_STRATUM])
        for s in range(STRATA)
    ]


def build_geometry(
    triplets: list[np.ndarray], use_role: bool, use_strata: bool
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    units: list[tuple[str, np.ndarray]] = []
    role_records: list[tuple[np.ndarray, np.ndarray]] = []
    for expert, source in enumerate(triplets):
        if use_role:
            channels, basis, eigenvalues = role_transform(source)
            role_records.append((basis, eigenvalues))
        else:
            channels = source
        for channel in range(ROLES):
            matrix = channels[channel]
            if use_strata:
                for label, index in enumerate(strata(matrix)):
                    units.append(
                        (
                            f"expert_{expert:02d}_channel_{channel}_stratum_{label}",
                            np.ascontiguousarray(matrix[index], dtype=np.float64),
                        )
                    )
            else:
                units.append((f"expert_{expert:02d}_channel_{channel}", matrix))
    return units, role_records


def polar_curve(matrix: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n, m = matrix.shape
    singular = np.sort(np.linalg.svd(matrix, compute_uv=False).astype(np.float64))
    energy = float(np.sum(np.square(matrix), dtype=np.float64))
    close(np.dot(singular, singular), energy, rtol=2e-10, atol=2e-9)
    prefix = np.concatenate(([0.0], np.cumsum(singular, dtype=np.float64)))
    prefix2 = np.concatenate(([0.0], np.cumsum(np.square(singular), dtype=np.float64)))
    stiefel = n * m - n * (n + 1) // 2
    result: list[dict[str, Any]] = []
    for rank in range(n - 1):
        width = n - rank
        sums = prefix[width:] - prefix[:-width]
        sums2 = prefix2[width:] - prefix2[:-width]
        errors = np.maximum(0.0, sums2 - sums * sums / width)
        start = int(np.argmin(errors))
        model_dof = stiefel + 1 + n * rank - rank * (rank - 1) // 2
        normal_dof = n * m - model_dof
        residual = float(errors[start])
        if residual <= 0.0 or energy - residual <= 0.0:
            continue
        result.append(
            {
                "rank": rank,
                "window_start": start,
                "window_stop": start + width,
                "common_scale": float(sums[start] / width),
                "model_dof": model_dof,
                "normal_dof": normal_dof,
                "model_energy": energy - residual,
                "normal_energy": residual,
            }
        )
    return singular, result


def compare_curve(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    if len(actual) != len(expected):
        raise AssertionError((len(actual), len(expected)))
    integers = ("rank", "window_start", "window_stop", "model_dof", "normal_dof")
    floats = ("common_scale", "model_energy", "normal_energy")
    for a, e in zip(actual, expected, strict=True):
        for key in integers:
            if int(a[key]) != int(e[key]):
                raise AssertionError((key, a[key], e[key]))
        for key in floats:
            close(a[key], e[key], rtol=8e-10, atol=8e-9)


def waterfill(d: np.ndarray, e: np.ndarray, rate: float) -> dict[str, float | int]:
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    logv = np.log2(e / d)
    order = np.argsort(logv)[::-1]
    lv = logv[order]
    ds = d[order]
    cd = np.cumsum(ds)
    cdlv = np.cumsum(ds * lv)
    levels = (cdlv - 2.0 * rate) / cd
    active_count = len(d)
    for k in range(1, len(d) + 1):
        level = levels[k - 1]
        if level <= lv[k - 1] + 2e-14 and (k == len(d) or level >= lv[k] - 2e-14):
            active_count = k
            break
    log_level = float(levels[active_count - 1])
    level = 2.0**log_level
    active = logv > log_level
    used = float(np.sum(d * 0.5 * np.maximum(0.0, logv - log_level)))
    close(used, rate, rtol=2e-10, atol=2e-10)
    distortion = float(np.sum(np.where(active, d * level, e)))
    return {
        "distortion": distortion,
        "active": int(np.count_nonzero(active)),
        "level": level,
    }


def expected_side(modules: dict[str, bool], units: list[tuple[str, np.ndarray]]) -> dict[str, int]:
    role = EXPERTS * 3 * 16 if modules["role_gauge"] else 0
    labels = MATRICES * ROWS * 3 if modules["strata"] else 0
    polar = (
        sum(2 * math.ceil(math.log2(matrix.shape[0])) for _, matrix in units)
        if modules["polar"]
        else 0
    )
    return {
        "global_header": GLOBAL_HEADER_BITS,
        "role_klt_q15_angles": role,
        "strata_uint3_labels": labels,
        "polar_rank_and_window_labels": polar,
        "total": GLOBAL_HEADER_BITS + role + labels + polar,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    result_bytes = args.result.read_bytes()
    result = json.loads(result_bytes)
    claimed = result.get("result_lock_sha256")
    clean = dict(result)
    clean.pop("result_lock_sha256", None)
    if hashlib.sha256(canonical(clean)).hexdigest() != claimed:
        raise AssertionError("result internal seal")
    if result["schema"] != "qwen_nested_composite_superoracle_v1":
        raise AssertionError("schema")
    _, _, matrices = load_sources(args.root.resolve(), result)
    triplets = [np.stack(matrices[3 * i : 3 * i + 3], axis=0) for i in range(EXPERTS)]
    source_energy = float(sum(np.sum(np.square(x), dtype=np.float64) for x in matrices))
    close(source_energy, result["scope"]["source_energy"], rtol=2e-13, atol=2e-9)

    rebuilt: dict[str, list[tuple[str, np.ndarray]]] = {}
    for key, geometry in result["geometries"].items():
        modules = geometry["modules"]
        units, role_records = build_geometry(
            triplets, bool(modules["role_gauge"]), bool(modules["strata"])
        )
        rebuilt[key] = units
        if len(units) != int(geometry["unit_count"]):
            raise AssertionError((key, "unit count"))
        if modules["role_gauge"]:
            if len(role_records) != len(geometry["role_records"]):
                raise AssertionError("role records")
            for (basis, eigenvalues), recorded in zip(
                role_records, geometry["role_records"], strict=True
            ):
                if not np.allclose(basis, recorded["basis"], rtol=3e-11, atol=3e-11):
                    raise AssertionError("role basis")
                if not np.allclose(
                    eigenvalues,
                    recorded["role_energy_eigenvalues"],
                    rtol=3e-11,
                    atol=3e-8,
                ):
                    raise AssertionError("role eigenvalues")
        plain_energy = 0.0
        for ordinal, ((name, matrix), plain, polar) in enumerate(
            zip(units, geometry["plain_components"], geometry["polar_units"], strict=True)
        ):
            if name != plain["name"] or name != polar["name"]:
                raise AssertionError((key, ordinal, "unit name"))
            if int(plain["dimension"]) != matrix.size:
                raise AssertionError((key, ordinal, "dimension"))
            energy = float(np.sum(np.square(matrix), dtype=np.float64))
            close(energy, plain["energy"], rtol=3e-12, atol=3e-9)
            plain_energy += energy
            singular, curve = polar_curve(matrix)
            if not np.allclose(
                singular,
                np.asarray(polar["singular_values"], dtype=np.float64),
                rtol=8e-10,
                atol=8e-10,
            ):
                raise AssertionError((key, ordinal, "singular spectrum"))
            compare_curve(curve, polar["rank_curve"])
        close(plain_energy, source_energy, rtol=3e-13, atol=3e-8)
        print(f"verified geometry {key} ({len(units)} units)", flush=True)

    charged: list[dict[str, Any]] = []
    leaky: list[dict[str, Any]] = []
    for variant, variant_row in result["variants"].items():
        geometry = result["geometries"][variant_row["geometry"]]
        units = rebuilt[variant_row["geometry"]]
        modules = variant_row["modules"]
        side = expected_side(modules, units)
        if side != variant_row["explicit_side_bits"]:
            raise AssertionError((variant, "side ledger"))
        for rate_text, score in variant_row["rates"].items():
            rate = float(rate_text)
            if modules["polar"]:
                dims: list[float] = []
                energies: list[float] = []
                model_d: list[float] = []
                model_e: list[float] = []
                normal_d: list[float] = []
                normal_e: list[float] = []
                if len(score["selections"]) != len(units):
                    raise AssertionError((variant, rate, "selection count"))
                for ordinal, selection in enumerate(score["selections"]):
                    curve = geometry["polar_units"][ordinal]["rank_curve"]
                    row = curve[int(selection["rank_curve_index"])]
                    for field in ("rank", "model_dof", "normal_dof"):
                        if int(row[field]) != int(selection[field]):
                            raise AssertionError((variant, rate, ordinal, field))
                    dims.extend((row["model_dof"], row["normal_dof"]))
                    energies.extend((row["model_energy"], row["normal_energy"]))
                    model_d.append(row["model_dof"] / PANEL_VALUES)
                    model_e.append(row["model_energy"] / source_energy)
                    normal_d.append(row["normal_dof"] / PANEL_VALUES)
                    normal_e.append(row["normal_energy"] / source_energy)
            else:
                dims = [row["dimension"] for row in geometry["plain_components"]]
                energies = [row["energy"] for row in geometry["plain_components"]]
            d = np.asarray(dims, dtype=np.float64) / PANEL_VALUES
            e = np.asarray(energies, dtype=np.float64) / source_energy
            close(np.sum(d), 1.0, rtol=0.0, atol=2e-15)
            close(np.sum(e), 1.0, rtol=3e-13, atol=3e-13)
            payload = rate - side["total"] / PANEL_VALUES
            wf = waterfill(d, e, payload)
            f_value = float(wf["distortion"]) * 2.0 ** (2.0 * rate)
            close(f_value, score["F"], rtol=4e-10, atol=4e-10)
            close(wf["distortion"], score["ideal_relative_mse"], rtol=4e-10, atol=4e-10)
            if bool(f_value <= TARGET_F) != bool(score["passes_F_le_0p8"]):
                raise AssertionError((variant, rate, "decision"))
            expected_cold = math.ceil(PANEL_VALUES * rate / 8 / EXPERTS) + 4096
            if expected_cold != int(score["read_ledger"]["cold_expert_bytes_with_4KiB_manifest"]):
                raise AssertionError((variant, rate, "read bytes"))
            if f_value <= TARGET_F:
                charged.append(
                    {
                        "variant": variant,
                        "rate": rate,
                        "F": f_value,
                        "module_count": sum(bool(x) for x in modules.values()),
                        "side_bpw": side["total"] / PANEL_VALUES,
                    }
                )
            if modules["polar"]:
                envelopes = score["source_leaky_envelopes"]
                pairs = (
                    (
                        "free_manifold_predictor_encode_normal_only",
                        np.asarray(normal_d),
                        np.asarray(normal_e),
                        sum(model_d),
                    ),
                    (
                        "free_normal_correction_encode_manifold_only",
                        np.asarray(model_d),
                        np.asarray(model_e),
                        sum(normal_d),
                    ),
                )
                for name, ld, le, free_dof in pairs:
                    lwf = waterfill(ld, le, rate)
                    lf = float(lwf["distortion"]) * 2.0 ** (2.0 * rate)
                    close(lf, envelopes[name]["F"], rtol=4e-10, atol=4e-10)
                    close(free_dof, envelopes[name]["free_side_dof_fraction"])
                    if lf <= TARGET_F:
                        leaky.append(
                            {
                                "variant": variant,
                                "envelope": name,
                                "rate": rate,
                                "F": lf,
                                "module_count": sum(bool(x) for x in modules.values()),
                                "free_side_dof_fraction": free_dof,
                                "fp16_side_rate_bpw": 16.0 * free_dof,
                            }
                        )
        print(f"verified scores {variant}", flush=True)

    charged.sort(key=lambda x: (x["module_count"], x["F"], x["side_bpw"]))
    leaky.sort(key=lambda x: (x["module_count"], x["fp16_side_rate_bpw"], x["F"]))
    decision = result["decision"]
    if bool(charged) != bool(decision["charged_target_reached"]):
        raise AssertionError("charged aggregate decision")
    if bool(leaky) != bool(decision["source_leaky_target_reached"]):
        raise AssertionError("leaky aggregate decision")
    if charged:
        expected = decision["minimal_charged_pass"]
        if charged[0]["variant"] != expected["variant"] or charged[0]["rate"] != expected["rate"]:
            raise AssertionError("minimal charged pass")
    if leaky:
        expected = decision["minimal_source_leaky_pass"]
        if (
            leaky[0]["variant"] != expected["variant"]
            or leaky[0]["envelope"] != expected["envelope"]
            or leaky[0]["rate"] != expected["rate"]
        ):
            raise AssertionError("minimal leaky pass")

    receipt = {
        "schema": "qwen_nested_composite_superoracle_verification_v1",
        "passed": True,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "result_internal_lock_sha256": claimed,
        "all_18_source_hashes_rechecked": True,
        "all_four_geometries_rebuilt": True,
        "all_324_singular_spectra_recomputed": True,
        "all_polar_curves_recomputed": True,
        "all_joint_waterfills_recomputed": True,
        "charged_target_reached": bool(charged),
        "source_leaky_target_reached": bool(leaky),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"PASS: wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
