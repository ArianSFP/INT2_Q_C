#!/usr/bin/env python3
"""Source-only capped LOGIC-Q v1 adapter.

The module contains no checkpoint/file-format adapter and never imports CuPy.
Production callers inject an ``xp`` module; a live pilot must inject CuPy.
The exact audited v0 package is a hash-pinned packet-mechanics dependency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PARENT_MANIFEST_SHA256 = "31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced"
PARENT_SOURCE_ROOT_SHA256 = "2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a"
ADAPTER_SCHEMA = "logic-q-label-flexible-algebraic-gate-v1-capped-adapter"
SPLIT_DOMAIN = b"logic-q-v1-capped-adapter-split\0"
CONTROL_DOMAIN = b"logic-q-v1-capped-adapter-control\0"
CONTROL_SEEDS = (10619863, 10619881, 10619909, 10619927,
                 10619953, 10619971, 10619999, 10620017)
ROLE_ORDER = ("gate", "up", "down_transposed")
ROLE_ORDINAL = {role: index for index, role in enumerate(ROLE_ORDER)}
TARGET_F = 0.8
RATE_MIN = 2.15
RATE_MAX = 2.5


class AdapterError(RuntimeError):
    """Fail-closed source, packet, cap, or protocol error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               AdapterError(f"{label} nonfinite JSON {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AdapterError(f"{label} unavailable") from exc
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AdapterError(f"{label} read") from exc
    after = path.lstat()
    require((after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino) ==
            (info.st_size, info.st_mtime_ns, info.st_mode, info.st_ino),
            f"{label} changed during read")
    return payload


def verify_parent_package(package: Path) -> dict[str, Any]:
    """Authenticate every member of the independently audited v0 package."""
    root = package.resolve(strict=True)
    require(root.is_dir(), "parent package directory")
    manifest_payload = _regular_bytes(root / "SOURCE_MANIFEST.json", "parent manifest")
    require(sha256(manifest_payload) == PARENT_MANIFEST_SHA256,
            "parent manifest external pin")
    manifest = _strict_json(manifest_payload, "parent manifest")
    require(manifest.get("source_root_sha256") == PARENT_SOURCE_ROOT_SHA256,
            "parent source root external pin")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "parent members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "parent member schema")
        name = row["name"]
        require(isinstance(name, str) and name and "/" not in name and "\\" not in name
                and name != "SOURCE_MANIFEST.json" and name not in names,
                "parent safe member")
        payload = _regular_bytes(root / name, f"parent member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"parent member pin {name}")
        observed.append(item)
        names.append(name)
    parent_root_payload = json.dumps(
        observed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    require(sha256(parent_root_payload) == PARENT_SOURCE_ROOT_SHA256,
            "parent observed root")
    actual = {entry.name for entry in os.scandir(root)}
    require(actual == set(names) | {"SOURCE_MANIFEST.json"},
            "parent exact member closure")
    return {"manifest_sha256": PARENT_MANIFEST_SHA256,
            "source_root_sha256": PARENT_SOURCE_ROOT_SHA256,
            "members": tuple(names)}


def load_parent_core(package: Path) -> Any:
    verify_parent_package(package)
    path = package.resolve(strict=True) / "logicq_core.py"
    module_name = "logicq_v0_pinned_" + PARENT_SOURCE_ROOT_SHA256[:16]
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "parent import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PilotConfig:
    config_id: str
    block_size: int
    profile_bank: tuple[int, ...]
    scale_shortlist: int
    normalized_lambda: float
    rm_word_shortlist: int
    rm_pair_keep: int
    rm_exception_cap: int
    rm_batch_blocks: int
    romdd_depths: tuple[int, ...]
    romdd_exception_cap: int


FROZEN_CONFIGS = (
    PilotConfig("b256-l005", 256, (0, 1), 2, 0.005, 8, 4, 16, 64,
                (0, 2, 4, 6), 64),
    PilotConfig("b256-l010", 256, (0, 1), 2, 0.010, 8, 4, 16, 64,
                (0, 2, 4, 6), 64),
    PilotConfig("b256-l020", 256, (0, 1), 2, 0.020, 8, 4, 16, 64,
                (0, 2, 4, 6), 64),
)

MAX_COMPONENT_WEIGHTS = 2_097_152
MAX_BLOCK_SIZE = 256
MAX_SCALE_SHORTLIST = 2
MAX_RM_WORD_SHORTLIST = 8
MAX_RM_PAIR_KEEP = 4
MAX_RM_EXCEPTIONS = 16
MAX_ROMDD_DEPTH = 6
MAX_ROMDD_EXCEPTIONS = 64


def config_record(config: PilotConfig) -> dict[str, Any]:
    return asdict(config)


def frozen_grid_record() -> dict[str, Any]:
    rows = [config_record(config) for config in FROZEN_CONFIGS]
    return {"schema": "logic-q-v1-frozen-grid", "configs": rows,
            "sha256": sha256(canonical_json(rows))}


def validate_config(config: PilotConfig, rows: int, cols: int) -> None:
    n = int(rows) * int(cols)
    require(config in FROZEN_CONFIGS, "config belongs to frozen grid")
    require(0 < rows and 0 < cols and n <= MAX_COMPONENT_WEIGHTS,
            "component pilot size cap")
    require(config.block_size <= MAX_BLOCK_SIZE and n % config.block_size == 0
            and config.block_size & (config.block_size - 1) == 0,
            "capped block geometry")
    require(1 <= config.scale_shortlist <= MAX_SCALE_SHORTLIST and
            1 <= config.rm_word_shortlist <= MAX_RM_WORD_SHORTLIST and
            1 <= config.rm_pair_keep <= MAX_RM_PAIR_KEEP and
            0 <= config.rm_exception_cap <= MAX_RM_EXCEPTIONS and
            1 <= config.rm_batch_blocks <= 64,
            "RM frozen caps")
    require(config.romdd_depths and
            max(config.romdd_depths) <= MAX_ROMDD_DEPTH and
            config.romdd_exception_cap <= MAX_ROMDD_EXCEPTIONS,
            "ROMDD frozen caps")
    require(config.profile_bank and len(config.profile_bank) <= 2 and
            all(0 <= value < 4 for value in config.profile_bank),
            "profile frozen caps")


def _to_numpy(np: Any, value: Any) -> Any:
    if getattr(np, "__name__", "") == "numpy":
        return np.asarray(value)
    if hasattr(np, "asnumpy"):
        return np.asnumpy(value)
    if hasattr(value, "get"):
        return value.get()
    raise AdapterError("accelerator backend has no host transfer")


def _host_numpy(xp: Any) -> Any:
    if getattr(xp, "__name__", "") == "numpy":
        return xp
    module = importlib.import_module("numpy")
    require(getattr(module, "__name__", "") == "numpy", "host NumPy module")
    return module


def require_live_cupy(xp: Any, live: bool) -> None:
    if live:
        require(getattr(xp, "__name__", "") == "cupy",
                "live pilot requires injected CuPy backend")


def _bf16_words_for_rms(core: Any, rms: Sequence[float]) -> list[list[int]]:
    result = []
    for value in rms:
        words = []
        for exponent in range(-12, 13):
            words.append(core.fp32_to_bf16_bits(float(value) *
                                                2.0 ** (exponent / 8.0)))
        result.append(words)
    return result


def scale_shortlists(xp: Any, core: Any, values: Any, weights: Any, *,
                     block_size: int, profile: int, keep: int,
                     batch_blocks: int = 128) -> tuple[tuple[int, ...], ...]:
    """Frozen weighted-nearest shortlist; structured RM search scores every item."""
    source = xp.asarray(values, dtype=xp.float64).reshape(-1)
    importance = xp.asarray(weights, dtype=xp.float64).reshape(-1)
    require(source.shape == importance.shape and source.size % block_size == 0,
            "scale shortlist geometry")
    require(1 <= keep <= MAX_SCALE_SHORTLIST and 0 <= profile < 4,
            "scale shortlist cap")
    blocks = source.reshape(-1, block_size)
    weight_blocks = importance.reshape(-1, block_size)
    ratios = xp.asarray(core.PROFILE_RATIOS[profile], dtype=xp.float64)
    output: list[tuple[int, ...]] = []
    for start in range(0, int(blocks.shape[0]), batch_blocks):
        x = blocks[start:start + batch_blocks]
        w = weight_blocks[start:start + batch_blocks]
        rms = xp.sqrt(xp.maximum(xp.mean(x * x, axis=1), 2.0 ** -126))
        rms_host = [float(value) for value in _to_numpy(xp, rms).reshape(-1)]
        words = _bf16_words_for_rms(core, rms_host)
        decoded = [[core.bf16_bits_to_float(word) for word in row]
                   for row in words]
        scale_values = xp.asarray(decoded, dtype=xp.float64)
        levels = scale_values[:, :, None, None] * ratios[None, None, None, :]
        residual = x[:, None, :, None] - levels
        nearest = xp.min(residual * residual, axis=3)
        scores = xp.sum(w[:, None, :] * nearest, axis=2, dtype=xp.float64)
        scores_host = _to_numpy(xp, scores)
        for local, row_words in enumerate(words):
            order = sorted(range(len(row_words)),
                           key=lambda index: (float(scores_host[local, index]),
                                              int(row_words[index])))
            selected: list[int] = []
            for index in order:
                word = int(row_words[index])
                if word not in selected:
                    selected.append(word)
                if len(selected) == keep:
                    break
            require(len(selected) == keep, "distinct scale shortlist")
            output.append(tuple(selected))
    return tuple(output)


def _fast_exceptions(np: Any, core: Any, costs: Any, base_labels: Any,
                     lambda_per_bit: float, maximum: int, *,
                     fixed_prefix_bits: int) -> Any:
    """O(n log n + cap^2), never the v0 unrestricted O(n*maximum) path."""
    matrix = np.asarray(costs, dtype=np.float64)
    base = np.asarray(base_labels, dtype=np.uint8).reshape(-1)
    n = int(base.size)
    require(matrix.shape == (n, 4) and 0 <= maximum <= MAX_ROMDD_EXCEPTIONS,
            "capped exception geometry")
    indices = np.arange(n, dtype=np.int64)
    base_cost = matrix[indices, base.astype(np.int64)]
    masked = matrix.copy()
    masked[indices, base.astype(np.int64)] = np.inf
    alternatives = np.argmin(masked, axis=1).astype(np.uint8)
    alt_cost = masked[indices, alternatives.astype(np.int64)]
    deltas = alt_cost - base_cost
    cap = min(maximum, n)
    if cap:
        chosen = np.argpartition(deltas, cap - 1)[:cap]
        chosen_host = [int(value) for value in _to_numpy(np, chosen).reshape(-1)]
        delta_host = _to_numpy(np, deltas)
        order = sorted(chosen_host, key=lambda pos: (float(delta_host[pos]), pos))
    else:
        order = []
    base_total = float(_to_numpy(np, np.sum(base_cost, dtype=np.float64)))
    prefix = [0.0]
    for position in order:
        prefix.append(prefix[-1] + float(_to_numpy(np, deltas[position])))
    alt_host = _to_numpy(np, alternatives)
    best = None
    for count in range(cap + 1):
        positions = tuple(sorted(order[:count]))
        labels = tuple(int(alt_host[position]) for position in positions)
        distortion = base_total + prefix[count]
        bits = core.exception_bits(n, count)["total_bits"]
        charged = core.align_up(fixed_prefix_bits + bits, 8)
        objective = distortion + lambda_per_bit * charged
        candidate = core.ExceptionPlan(positions, labels, distortion, bits,
                                       charged, objective)
        key = (objective, charged, distortion, positions, labels)
        if best is None or key < best[0]:
            best = (key, candidate)
    require(best is not None, "capped exception candidate")
    return best[1]


def _rm_words(xp: Any, n: int) -> Any:
    require(n >= 2 and n <= MAX_BLOCK_SIZE and n & (n - 1) == 0,
            "RM capped word length")
    m = n.bit_length() - 1
    coordinates = xp.arange(n, dtype=xp.uint32)
    words = xp.empty((2 * n, n), dtype=xp.uint8)
    for message in range(2 * n):
        value = xp.full(n, message & 1, dtype=xp.uint8)
        for bit in range(m):
            if (message >> (bit + 1)) & 1:
                value ^= ((coordinates >> bit) & 1).astype(xp.uint8)
        words[message] = value
    return words


def _rm_plan_for_costs(xp: Any, core: Any, costs: Any, *,
                       lambda_per_bit: float, word_shortlist: int,
                       pair_keep: int, exception_cap: int,
                       words: Any) -> Any:
    matrix = xp.asarray(costs, dtype=xp.float64)
    n = int(matrix.shape[0])
    require(matrix.shape == (n, 4), "RM cost shape")
    bit_groups = (((0, 1), (2, 3)), ((0, 3), (1, 2)))
    shortlists = []
    for zeros, ones in bit_groups:
        zero_cost = xp.minimum(matrix[:, zeros[0]], matrix[:, zeros[1]])
        one_cost = xp.minimum(matrix[:, ones[0]], matrix[:, ones[1]])
        scores = xp.sum(zero_cost, dtype=xp.float64) + (one_cost - zero_cost) @ words.T
        score_host = _to_numpy(xp, scores).reshape(-1)
        order = sorted(range(score_host.size),
                       key=lambda index: (float(score_host[index]), index))
        shortlists.append(order[:word_shortlist])
    candidates = []
    for first in shortlists[0]:
        for second in shortlists[1]:
            labels = core.gray_labels_from_planes(
                xp, words[first], words[second]).reshape(-1)
            sse = core.weighted_sse_from_labels(xp, matrix, labels)
            candidates.append((sse, first, second, labels))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    m = n.bit_length() - 1
    base_bits = 2 * (m + 1)
    best = None
    for _, first, second, labels in candidates[:pair_keep]:
        plan = _fast_exceptions(
            xp, core, matrix, labels, lambda_per_bit, exception_cap,
            fixed_prefix_bits=base_bits)
        final = [int(value) for value in _to_numpy(xp, labels).reshape(-1)]
        for position, label in zip(plan.positions, plan.labels):
            final[position] = label
        candidate = core.RMBlockPlan(
            first, second, plan, tuple(final), base_bits,
            plan.charged_total_bits, plan.distortion, plan.objective,
            False, len(candidates))
        key = (candidate.objective, candidate.total_bits,
               candidate.distortion, first, second,
               candidate.exceptions.positions, candidate.exceptions.labels)
        if best is None or key < best[0]:
            best = (key, candidate)
    require(best is not None, "RM shortlisted plan")
    return best[1]


def _rm_batch_for_scales(xp: Any, host_np: Any, core: Any, source: Any,
                         importance: Any, scale_words: Sequence[int], *,
                         profile: int, words: Any, lambda_per_bit: float,
                         config: PilotConfig) -> tuple[list[Any], Any, Any]:
    """One batched CuPy shortlist/pair pass for one scale rank."""
    x = xp.asarray(source, dtype=xp.float64)
    w = xp.asarray(importance, dtype=xp.float64)
    batch, n = map(int, x.shape)
    require(w.shape == x.shape and n == config.block_size and
            len(scale_words) == batch, "batched RM geometry")
    ratios = xp.asarray(core.PROFILE_RATIOS[profile], dtype=xp.float64)
    decoded = [core.bf16_bits_to_float(int(word)) for word in scale_words]
    levels = xp.asarray(decoded, dtype=xp.float64)[:, None] * ratios[None, :]
    reconstruction = xp.broadcast_to(levels[:, None, :], (batch, n, 4))
    residual = x[:, :, None] - reconstruction
    costs = w[:, :, None] * residual * residual
    bit_groups = (((0, 1), (2, 3)), ((0, 3), (1, 2)))
    shortlists = []
    for zeros, ones in bit_groups:
        zero_cost = xp.minimum(costs[:, :, zeros[0]], costs[:, :, zeros[1]])
        one_cost = xp.minimum(costs[:, :, ones[0]], costs[:, :, ones[1]])
        scores = (xp.sum(zero_cost, axis=1, dtype=xp.float64)[:, None] +
                  (one_cost - zero_cost) @ words.T)
        score_host = _to_numpy(xp, scores)
        selected = host_np.empty((batch, config.rm_word_shortlist),
                                 dtype=host_np.int64)
        for row in range(batch):
            order = sorted(range(int(score_host.shape[1])),
                           key=lambda index: (float(score_host[row, index]), index))
            selected[row, :] = order[:config.rm_word_shortlist]
        shortlists.append(selected)
    first_words = words[xp.asarray(shortlists[0], dtype=xp.int64)]
    second_words = words[xp.asarray(shortlists[1], dtype=xp.int64)]
    bit_pairs = ((first_words[:, :, None, :] << 1) |
                 second_words[:, None, :, :])
    label_table = xp.asarray(core.BITS_TO_LABEL, dtype=xp.uint8)
    labels = label_table[bit_pairs].reshape(
        batch, config.rm_word_shortlist ** 2, n)
    gathered = xp.take_along_axis(
        costs[:, None, :, :], labels[:, :, :, None].astype(xp.int64), axis=3)
    pair_scores = xp.sum(gathered[:, :, :, 0], axis=2, dtype=xp.float64)
    pair_scores_host = _to_numpy(xp, pair_scores)
    labels_host = _to_numpy(xp, labels)
    costs_host = _to_numpy(xp, costs)
    reconstruction_host = _to_numpy(xp, reconstruction)
    plans = []
    m = n.bit_length() - 1
    base_bits = 2 * (m + 1)
    width = config.rm_word_shortlist
    for row in range(batch):
        order = sorted(range(width * width),
                       key=lambda index: (float(pair_scores_host[row, index]), index))
        best = None
        for flat in order[:config.rm_pair_keep]:
            first_local, second_local = divmod(flat, width)
            first = int(shortlists[0][row, first_local])
            second = int(shortlists[1][row, second_local])
            base = labels_host[row, flat]
            exceptions = _fast_exceptions(
                host_np, core, costs_host[row], base, lambda_per_bit,
                config.rm_exception_cap, fixed_prefix_bits=base_bits)
            final = [int(value) for value in base]
            for position, label in zip(exceptions.positions, exceptions.labels):
                final[position] = label
            candidate = core.RMBlockPlan(
                first, second, exceptions, tuple(final), base_bits,
                exceptions.charged_total_bits, exceptions.distortion,
                exceptions.objective, False, width * width)
            key = (candidate.objective, candidate.total_bits,
                   candidate.distortion, first, second,
                   candidate.exceptions.positions, candidate.exceptions.labels)
            if best is None or key < best[0]:
                best = (key, candidate)
        require(best is not None, "batched RM candidate")
        plans.append(best[1])
    return plans, costs_host, reconstruction_host


def _write_rm1_cached(writer: Any, host_np: Any, core: Any, words_host: Any,
                      n: int, plan: Any) -> None:
    width = n.bit_length()
    writer.write(int(plan.first_message), width)
    writer.write(int(plan.second_message), width)
    base = core.gray_labels_from_planes(
        host_np, words_host[plan.first_message],
        words_host[plan.second_message]).reshape(-1)
    core.write_exceptions(writer, n, tuple(int(value) for value in base),
                          plan.exceptions)
    writer.pad_to_byte()


def encode_rm1_capped(xp: Any, core: Any, values: Any, weights: Any, *,
                      role: str, rows: int, cols: int,
                      config: PilotConfig) -> Any:
    """Jointly score every frozen scale/RM-label pair without a full pair matrix."""
    validate_config(config, rows, cols)
    source = xp.asarray(values, dtype=xp.float64).reshape(-1)
    importance = xp.asarray(weights, dtype=xp.float64).reshape(-1)
    require(source.size == rows * cols and source.shape == importance.shape,
            "RM component source geometry")
    host_np = _host_numpy(xp)
    energy = float(_to_numpy(
        xp, xp.sum(importance * source * source, dtype=xp.float64)))
    require(math.isfinite(energy) and energy > 0.0, "RM source energy")
    lambda_per_bit = config.normalized_lambda * energy / int(source.size)
    words = _rm_words(xp, config.block_size)
    words_host = _to_numpy(xp, words)
    batched_cupy = getattr(xp, "__name__", "") == "cupy"
    best_profile = None
    for profile in config.profile_bank:
        shortlists = scale_shortlists(
            xp, core, source, importance, block_size=config.block_size,
            profile=profile, keep=config.scale_shortlist,
            batch_blocks=config.rm_batch_blocks)
        blocks = int(source.size) // config.block_size
        plans = []
        scales = []
        cost_rows = []
        reconstruction_rows = []
        if batched_cupy:
            source_blocks = source.reshape(blocks, config.block_size)
            weight_blocks = importance.reshape(blocks, config.block_size)
            for batch_start in range(0, blocks, config.rm_batch_blocks):
                batch_end = min(blocks, batch_start + config.rm_batch_blocks)
                runs = []
                for scale_index in range(config.scale_shortlist):
                    scale_words = [shortlists[index][scale_index]
                                   for index in range(batch_start, batch_end)]
                    runs.append(_rm_batch_for_scales(
                        xp, host_np, core,
                        source_blocks[batch_start:batch_end],
                        weight_blocks[batch_start:batch_end], scale_words,
                        profile=profile, words=words,
                        lambda_per_bit=lambda_per_bit, config=config))
                for local, block in enumerate(range(batch_start, batch_end)):
                    selected = None
                    for scale_index, (batch_plans, batch_costs,
                                      batch_reconstruction) in enumerate(runs):
                        plan = batch_plans[local]
                        word = int(shortlists[block][scale_index])
                        key = (plan.objective, plan.total_bits,
                               plan.distortion, word)
                        candidate = (key, word, plan, batch_costs[local],
                                     batch_reconstruction[local])
                        if selected is None or key < selected[0]:
                            selected = candidate
                    require(selected is not None, "batched joint scale/RM block")
                    _, word, plan, costs, reconstruction = selected
                    scales.append(word)
                    plans.append(plan)
                    cost_rows.append(costs)
                    reconstruction_rows.append(reconstruction)
        else:
            ratios = xp.asarray(core.PROFILE_RATIOS[profile], dtype=xp.float64)
            for block in range(blocks):
                start = block * config.block_size
                x = source[start:start + config.block_size]
                w = importance[start:start + config.block_size]
                block_best = None
                for word in shortlists[block]:
                    levels = core.bf16_bits_to_float(int(word)) * ratios
                    reconstruction = xp.broadcast_to(
                        levels[None, :], (config.block_size, 4)).copy()
                    residual = x[:, None] - reconstruction
                    costs = w[:, None] * residual * residual
                    plan = _rm_plan_for_costs(
                        xp, core, costs, lambda_per_bit=lambda_per_bit,
                        word_shortlist=config.rm_word_shortlist,
                        pair_keep=config.rm_pair_keep,
                        exception_cap=config.rm_exception_cap, words=words)
                    key = (plan.objective, plan.total_bits,
                           plan.distortion, int(word))
                    if block_best is None or key < block_best[0]:
                        block_best = (key, int(word), plan, costs,
                                      reconstruction)
                require(block_best is not None, "joint scale/RM block")
                _, word, plan, costs, reconstruction = block_best
                scales.append(word)
                plans.append(plan)
                cost_rows.append(costs)
                reconstruction_rows.append(reconstruction)
        objective = sum(plan.objective for plan in plans)
        bits = sum(plan.total_bits for plan in plans)
        distortion = sum(plan.distortion for plan in plans)
        key = (objective, bits, distortion, profile)
        if best_profile is None or key < best_profile[0]:
            best_profile = (key, profile, tuple(scales), tuple(plans),
                            host_np.concatenate(
                                [_to_numpy(xp, row) for row in cost_rows], axis=0),
                            host_np.concatenate(
                                [_to_numpy(xp, row) for row in reconstruction_rows],
                                axis=0))
    require(best_profile is not None, "RM capped profile")
    _, profile, scales, plans, costs, reconstruction = best_profile
    writer = core.BitWriter()
    labels = []
    for plan in plans:
        _write_rm1_cached(writer, host_np, core, words_host,
                          config.block_size, plan)
        labels.extend(plan.labels)
    source_host = _to_numpy(xp, source)
    importance_host = _to_numpy(xp, importance)
    component = core._build_component(
        host_np, family=core.FAMILY_RM1, role=role, rows=rows, cols=cols,
        block_size=config.block_size, parameter=1, profile=profile,
        scales=scales, writer=writer, labels=tuple(labels), values=source_host,
        weights=importance_host, costs=costs, reconstruction=reconstruction,
        exact_search=False, diagnostics={
            "search": "capped_plane_surrogate_shortlist_then_exact_four_level_pairs",
            "full_rm_pair_matrix_built": False,
            "word_shortlist": config.rm_word_shortlist,
            "pair_keep": config.rm_pair_keep,
            "exception_cap_per_block": config.rm_exception_cap,
            "scale_shortlist_per_block": config.scale_shortlist,
            "scale_and_labels_jointly_scored": True,
            "global_negative_authority": False,
            "cupy_backend": batched_cupy,
            "batched_accelerator_blocks": config.rm_batch_blocks,
        })
    return component


def _component_minimum_bytes(core: Any, family: int, rows: int, cols: int,
                             block_size: int, parameter: int) -> int:
    return core.ceil_div(core.family_minimum_component_bits(
        family, rows, cols, block_size, parameter), 8)


def _packed_bytes_from_component_lengths(core: Any, lengths: Sequence[int],
                                         weights: int) -> int:
    require(len(lengths) == 3, "three component lengths")
    cursor = core.EXPERT_HEADER_BYTES
    for length in lengths:
        cursor = core.align_up(cursor + int(length), core.COMPONENT_ALIGNMENT)
    base = core.align_up(cursor, core.EXPERT_PAGE)
    minimum = math.ceil(RATE_MIN * weights / 8.0)
    return max(base, core.align_up(minimum, core.EXPERT_PAGE))


def pooled_presearch_bound(core: Any, *, family: int, parameter: int,
                           rows: int, cols: int, block_size: int,
                           nearest_sse_by_role: Mapping[str, float],
                           energy_by_role: Mapping[str, float]) -> dict[str, Any]:
    """Executable uniform-family lower bound evaluated before family search."""
    require(set(nearest_sse_by_role) == set(ROLE_ORDER) and
            set(energy_by_role) == set(ROLE_ORDER), "presearch three roles")
    length = _component_minimum_bytes(core, family, rows, cols,
                                      block_size, parameter)
    # To prune this family from a paid per-role mixed bank, grant each other
    # role the shortest impossible-best component among every packet family.
    # This is strictly more optimistic than a uniform-family expert bound.
    other_lengths = (
        _component_minimum_bytes(core, core.FAMILY_LITERAL, rows, cols,
                                 block_size, 0),
        _component_minimum_bytes(core, core.FAMILY_RM1, rows, cols,
                                 block_size, 1),
        _component_minimum_bytes(core, core.FAMILY_GF2, rows, cols,
                                 block_size, 0),
        _component_minimum_bytes(core, core.FAMILY_ROMDD, rows, cols,
                                 block_size, 0),
    )
    other = min(other_lengths)
    weights = 3 * rows * cols
    physical_bytes = min(
        _packed_bytes_from_component_lengths(
            core, tuple(length if role == target else other for role in range(3)),
            weights)
        for target in range(3))
    physical_rate = physical_bytes * 8 / weights
    evaluation_rate = max(RATE_MIN, physical_rate)
    relative_mse = sum(nearest_sse_by_role.values()) / sum(energy_by_role.values())
    lower_f = relative_mse * 2.0 ** (2.0 * evaluation_rate)
    hard_kill = physical_rate > RATE_MAX or lower_f > TARGET_F
    if physical_rate > RATE_MAX:
        status = "HARD_KILL_BEFORE_SEARCH__MANDATORY_PACKED_EXPERT_RATE"
    elif lower_f > TARGET_F:
        status = "HARD_KILL_BEFORE_SEARCH__NEAREST_DISTORTION_MIN_RATE_F"
    else:
        status = "SURVIVES_PRESEARCH_BOUND"
    return {
        "family": core.FAMILY_NAMES[family], "parameter": parameter,
        "minimum_target_family_component_bytes": length,
        "minimum_impossible_best_other_component_bytes": other,
        "minimum_packed_expert_bytes": physical_bytes,
        "minimum_packed_rate_bpw": physical_rate,
        "nearest_pooled_relative_mse": relative_mse,
        "evaluation_rate_bpw": evaluation_rate,
        "optimistic_F_lower_bound": lower_f,
        "hard_kill": hard_kill, "status": status,
        "search_invoked": False,
        "bound_covers_target_family_in_any_paid_mixed_expert": True,
        "bound_grants_other_roles_impossible_best_family_components": True,
    }


def execute_if_survives(bound: Mapping[str, Any],
                        search: Callable[[], Any]) -> tuple[Any | None, dict[str, Any]]:
    require(set(("hard_kill", "search_invoked")) <= set(bound),
            "presearch bound schema")
    receipt = dict(bound)
    if bool(bound["hard_kill"]):
        receipt["search_invoked"] = False
        return None, receipt
    result = search()
    receipt["search_invoked"] = True
    return result, receipt


def _gf2_rref_pivots(np: Any, matrix: Any) -> tuple[Any, tuple[int, ...]]:
    value = np.asarray(matrix, dtype=np.uint8).copy()
    require(value.ndim == 2 and np.all(value < 2), "GF2 matrix")
    row = 0
    pivots = []
    for column in range(int(value.shape[1])):
        candidates = [index for index in range(row, int(value.shape[0]))
                      if int(value[index, column])]
        if not candidates:
            continue
        pivot = candidates[0]
        if pivot != row:
            value[[row, pivot]] = value[[pivot, row]]
        for index in range(int(value.shape[0])):
            if index != row and int(value[index, column]):
                value[index] ^= value[row]
        pivots.append(column)
        row += 1
        if row == int(value.shape[0]):
            break
    return value, tuple(pivots)


def _gf2_inverse(np: Any, matrix: Any) -> Any:
    value = np.asarray(matrix, dtype=np.uint8)
    require(value.ndim == 2 and value.shape[0] == value.shape[1],
            "GF2 inverse square")
    n = int(value.shape[0])
    augmented = np.concatenate((value.copy(), np.eye(n, dtype=np.uint8)), axis=1)
    for column in range(n):
        pivots = [row for row in range(column, n) if int(augmented[row, column])]
        require(pivots, "GF2 inverse singular")
        pivot = pivots[0]
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        for row in range(n):
            if row != column and int(augmented[row, column]):
                augmented[row] ^= augmented[column]
    require(np.array_equal(augmented[:, :n], np.eye(n, dtype=np.uint8)),
            "GF2 inverse closure")
    return augmented[:, n:]


def canonical_gf2_factors(np: Any, product: Any, declared_rank: int) -> tuple[Any, Any]:
    """Unique pivot-column factorization padded with deterministic zero gauge."""
    matrix = np.asarray(product, dtype=np.uint8)
    rows, cols = map(int, matrix.shape)
    _, pivot_columns = _gf2_rref_pivots(np, matrix)
    true_rank = len(pivot_columns)
    require(true_rank <= declared_rank <= min(rows, cols), "GF2 declared rank")
    left_true = matrix[:, pivot_columns] if true_rank else np.zeros((rows, 0), dtype=np.uint8)
    if true_rank:
        _, pivot_rows = _gf2_rref_pivots(np, left_true.T)
        require(len(pivot_rows) == true_rank, "GF2 canonical pivot rows")
        square = left_true[list(pivot_rows), :]
        inverse = _gf2_inverse(np, square)
        right_true = (inverse.astype(np.uint64) @
                      matrix[list(pivot_rows), :].astype(np.uint64) & 1).astype(np.uint8)
    else:
        right_true = np.zeros((0, cols), dtype=np.uint8)
    left = np.zeros((rows, declared_rank), dtype=np.uint8)
    right = np.zeros((declared_rank, cols), dtype=np.uint8)
    left[:, :true_rank] = left_true
    right[:true_rank, :] = right_true
    require(np.array_equal((left.astype(np.uint64) @ right.astype(np.uint64) & 1)
                           .astype(np.uint8), matrix), "GF2 canonical product")
    return left, right


def _canonicalize_gf2_payload(np: Any, core: Any, record: Any,
                              payload: bytes) -> bytes:
    reader = core.BitReader(payload, record.payload_bits)
    rank = int(record.parameter)
    shapes = ((record.rows, rank), (rank, record.cols),
              (record.rows, rank), (rank, record.cols))
    matrices = []
    for shape in shapes:
        bits = [reader.read(1) for _ in range(shape[0] * shape[1])]
        matrices.append(np.asarray(bits, dtype=np.uint8).reshape(shape))
    products = (core.gf2_product(np, matrices[0], matrices[1]),
                core.gf2_product(np, matrices[2], matrices[3]))
    canonical = (*canonical_gf2_factors(np, products[0], rank),
                 *canonical_gf2_factors(np, products[1], rank))
    writer = core.BitWriter()
    for matrix in canonical:
        for value in matrix.reshape(-1):
            writer.write(int(value), 1)
    while reader.position < reader.bit_length:
        writer.write(reader.read(1), 1)
    reader.finish()
    rewritten = writer.finish()
    require(writer.bit_length == record.payload_bits, "GF2 payload bit closure")
    return rewritten


def _romdd_canonical_depth(core: Any, record: Any, payload: bytes) -> int:
    reader = core.BitReader(payload, record.payload_bits)
    radices = core.coordinate_radices(record.rows, record.cols)
    node_count = reader.read(32)
    reference_width = core.ceil_log2_count(4 + node_count)
    level_width = core.ceil_log2_count(max(1, len(radices)))
    root = reader.read(reference_width)
    require(root < 4 + node_count, "ROMDD root")
    levels = []
    for index in range(node_count):
        level = reader.read(level_width)
        require(level < len(radices), "ROMDD level")
        children = [reader.read(reference_width) for _ in range(radices[level])]
        require(all(child < 4 + index for child in children),
                "ROMDD topological children")
        levels.append(level)
    return 0 if not levels else 1 + max(levels)


def canonicalize_component(np: Any, core: Any, packet: bytes) -> bytes:
    record, scales, payload = core.parse_component_envelope(bytes(packet))
    new_record = record
    new_payload = payload
    if record.family == core.FAMILY_GF2:
        new_payload = _canonicalize_gf2_payload(np, core, record, payload)
    elif record.family == core.FAMILY_ROMDD:
        depth = _romdd_canonical_depth(core, record, payload)
        new_record = replace(record, parameter=depth)
    result = core.component_packet(new_record, scales, new_payload)
    old_decoded = core.decode_component(np, packet)[0]
    new_decoded = core.decode_component(np, result)[0]
    require(old_decoded == new_decoded, "canonical component label invariant")
    require(result == core.component_packet(
        core.parse_component_envelope(result)[0],
        core.parse_component_envelope(result)[1],
        core.parse_component_envelope(result)[2]),
        "canonical component exact envelope re-encode")
    return result


def decode_canonical_component(np: Any, core: Any, packet: bytes) -> Any:
    require(bytes(packet) == canonicalize_component(np, core, packet),
            "noncanonical component packet")
    return core.decode_component(np, packet)


def pack_canonical_expert(np: Any, core: Any,
                          components: Mapping[str, bytes]) -> bytes:
    require(set(components) == set(ROLE_ORDER), "exact SwiGLU roles")
    canonical: dict[str, bytes] = {}
    shapes = []
    for role in ROLE_ORDER:
        packet = canonicalize_component(np, core, bytes(components[role]))
        record, _, _ = core.parse_component_envelope(packet)
        require(record.role == role, "canonical role identity")
        shapes.append((record.rows, record.cols))
        canonical[role] = packet
    require(len(set(shapes)) == 1,
            "Gate/Up/DownT canonical SwiGLU shape equality")
    packet = core.pack_expert(canonical)
    weights = 3 * shapes[0][0] * shapes[0][1]
    target_bytes = _packed_bytes_from_component_lengths(
        core, [len(canonical[role]) for role in ROLE_ORDER], weights)
    require(target_bytes >= len(packet) and target_bytes % core.EXPERT_PAGE == 0,
            "canonical expert target pages")
    packet += b"\0" * (target_bytes - len(packet))
    return packet


def _expert_component_slices(core: Any, packet: bytes) -> dict[str, bytes]:
    require(len(packet) >= core.EXPERT_HEADER_BYTES, "expert envelope")
    fields = core.EXPERT_HEADER.unpack(packet[:core.EXPERT_HEADER_BYTES])
    magic, version, count, reserved16 = fields[:4]
    lengths = fields[4:7]
    reserved = fields[7]
    require(magic == core.EXPERT_MAGIC and version == 0 and count == 3 and
            reserved16 == 0 and reserved == b"\0" * 28, "expert header")
    cursor = core.EXPERT_HEADER_BYTES
    result = {}
    for role, length in zip(ROLE_ORDER, lengths):
        require(length >= core.COMPONENT_HEADER_BYTES and
                cursor + length <= len(packet), "expert component span")
        result[role] = packet[cursor:cursor + length]
        cursor = core.align_up(cursor + length, core.COMPONENT_ALIGNMENT)
    require(packet[cursor:] == b"\0" * (len(packet) - cursor),
            "expert trailing zero pages")
    return result


def unpack_canonical_expert(np: Any, core: Any, packet: bytes) -> Any:
    parts = _expert_component_slices(core, bytes(packet))
    records = []
    for role in ROLE_ORDER:
        require(parts[role] == canonicalize_component(np, core, parts[role]),
                "expert component canonical bytes")
        record = core.parse_component_envelope(parts[role])[0]
        require(record.role == role, "expert role order")
        records.append(record)
    require(len({(record.rows, record.cols) for record in records}) == 1,
            "expert exact SwiGLU shape closure")
    decoded = core.unpack_expert(np, packet)
    require(packet == pack_canonical_expert(np, core, parts),
            "expert canonical re-encode")
    return decoded


def pooled_expert_score(np: Any, core: Any, packet: bytes,
                        components: Mapping[str, Any]) -> dict[str, Any]:
    unpack_canonical_expert(np, core, packet)
    require(set(components) == set(ROLE_ORDER), "pooled score roles")
    weighted_sse = sum(float(components[role].weighted_sse) for role in ROLE_ORDER)
    source_energy = sum(float(components[role].source_energy) for role in ROLE_ORDER)
    weights = sum(len(components[role].labels) for role in ROLE_ORDER)
    rate = len(packet) * 8 / weights
    relative_mse = weighted_sse / source_energy
    f_value = relative_mse * 2.0 ** (2.0 * rate)
    read_bytes = len(packet)
    return {
        "physical_expert_bytes": len(packet), "expert_weights": weights,
        "physical_rate_bpw": rate, "weighted_sse": weighted_sse,
        "source_energy": source_energy, "pooled_relative_mse": relative_mse,
        "F": f_value, "rate_interval_pass": RATE_MIN <= rate <= RATE_MAX,
        "target_F_pass": f_value <= TARGET_F,
        "routed_storage_read_bytes": read_bytes,
        "read_passes": 1,
        "cold_read_amplification": read_bytes / len(packet),
        "cold_read_below_2x": read_bytes / len(packet) < 2.0,
        "component_or_per_role_gate_used": False,
        "all_headers_alignment_and_final_page_charged": True,
    }


def _nearest_components(xp: Any, core: Any, roles: Mapping[str, tuple[Any, Any]],
                        rows: int, cols: int, config: PilotConfig) -> dict[str, Any]:
    result = {}
    for role in ROLE_ORDER:
        values, weights = roles[role]
        result[role] = core.encode_literal_component(
            xp, values, weights, role=role, rows=rows, cols=cols,
            block_size=config.block_size)
    return result


def _encode_romdd_capped(xp: Any, core: Any, values: Any, weights: Any, *,
                         role: str, rows: int, cols: int,
                         config: PilotConfig, lambda_per_bit: float) -> Any:
    candidates = [core.encode_romdd_component(
        xp, values, weights, role=role, rows=rows, cols=cols,
        block_size=config.block_size, lambda_per_bit=lambda_per_bit,
        depths=config.romdd_depths, profile=profile,
        exception_limit=config.romdd_exception_cap)
        for profile in config.profile_bank]
    require(len(candidates) <= 2, "ROMDD profile cap")
    return min(candidates, key=lambda component: (
        component.weighted_sse + lambda_per_bit * component.physical_bits,
        component.physical_bits, component.weighted_sse, component.family))


def encode_expert(xp: Any, core: Any,
                  roles: Mapping[str, tuple[Any, Any]], *, rows: int, cols: int,
                  config: PilotConfig, live: bool = False) -> dict[str, Any]:
    """Execute the capped bank and one packed-expert gate.

    GF2 search is deliberately absent: v1 only canonicalizes/decodes such
    packets. ROMDD is a bounded negative because its scales are nearest-first.
    """
    require_live_cupy(xp, live)
    validate_config(config, rows, cols)
    require(set(roles) == set(ROLE_ORDER), "expert exact input roles")
    host_np = _host_numpy(xp)
    host_roles = {role: (_to_numpy(xp, roles[role][0]),
                         _to_numpy(xp, roles[role][1]))
                  for role in ROLE_ORDER}
    literal = _nearest_components(host_np, core, host_roles,
                                  rows, cols, config)
    nearest_sse = {role: literal[role].weighted_sse for role in ROLE_ORDER}
    energies = {role: literal[role].source_energy for role in ROLE_ORDER}
    bounds = {}
    family_components: dict[str, dict[str, Any]] = {"literal4": literal}

    rm_bound = pooled_presearch_bound(
        core, family=core.FAMILY_RM1, parameter=1, rows=rows, cols=cols,
        block_size=config.block_size, nearest_sse_by_role=nearest_sse,
        energy_by_role=energies)
    rm_result, bounds["rm1_plus_exceptions"] = execute_if_survives(
        rm_bound, lambda: {role: encode_rm1_capped(
            xp, core, roles[role][0], roles[role][1], role=role,
            rows=rows, cols=cols, config=config) for role in ROLE_ORDER})
    if rm_result is not None:
        family_components["rm1_plus_exceptions"] = rm_result

    romdd_bound = pooled_presearch_bound(
        core, family=core.FAMILY_ROMDD, parameter=0, rows=rows, cols=cols,
        block_size=config.block_size, nearest_sse_by_role=nearest_sse,
        energy_by_role=energies)
    energy = sum(energies.values())
    lam = config.normalized_lambda * energy / (3 * rows * cols)
    romdd_result, bounds["romdd_plus_exceptions"] = execute_if_survives(
        romdd_bound, lambda: {role: _encode_romdd_capped(
            host_np, core, host_roles[role][0], host_roles[role][1], role=role,
            rows=rows, cols=cols, config=config, lambda_per_bit=lam)
            for role in ROLE_ORDER})
    if romdd_result is not None:
        family_components["romdd_plus_exceptions"] = romdd_result

    # Select a family independently per paid component; selectors are in headers.
    chosen = {}
    for role in ROLE_ORDER:
        candidates = [bank[role] for bank in family_components.values()]
        chosen[role] = min(candidates, key=lambda component: (
            component.weighted_sse + lam * component.physical_bits,
            component.physical_bits, component.weighted_sse, component.family))
    canonical_packets = {role: canonicalize_component(
        host_np, core, chosen[role].packet) for role in ROLE_ORDER}
    packet = pack_canonical_expert(host_np, core, canonical_packets)
    score = pooled_expert_score(host_np, core, packet, chosen)
    score.update({
        "schema": ADAPTER_SCHEMA + "-expert-result",
        "config_id": config.config_id,
        "selected_family_by_role": {role: chosen[role].family for role in ROLE_ORDER},
        "presearch": bounds,
        "gf2_search": "NOT_SCHEDULED__NO_QWEN_SCALE_BOUNDED_SEARCH__NO_NEGATIVE_AUTHORITY",
        "romdd_scale_search": "NEAREST_SCALE_CONDITIONED_BOUNDED_NEGATIVE_ONLY",
        "rm_scale_search": "FROZEN_SHORTLIST_JOINT_WITH_LABELS",
        "live_cupy_required_and_used": bool(live),
    })
    return {"packet": packet, "components": chosen, "score": score}


@dataclass(frozen=True)
class PanelRow:
    layer: str
    slot: str
    role: str
    rows: int
    cols: int
    source_sha256: str


def _public_hash(kind: str, value: str) -> bytes:
    return hashlib.sha256(SPLIT_DOMAIN + kind.encode("ascii") + b"\0" +
                          value.encode("utf-8")).digest()


def canonical_panel(rows: Iterable[PanelRow]) -> tuple[PanelRow, ...]:
    panel = tuple(rows)
    require(panel, "nonempty panel")
    require(all(row.role in ROLE_ORDINAL and row.layer and row.slot and
                row.rows > 0 and row.cols > 0 and len(row.source_sha256) == 64 and
                all(char in "0123456789abcdef" for char in row.source_sha256)
                for row in panel), "panel row fields")
    ordered = tuple(sorted(panel, key=lambda row: (
        row.layer.encode("utf-8"), row.slot.encode("utf-8"), ROLE_ORDINAL[row.role])))
    keys = [(row.layer, row.slot, row.role) for row in ordered]
    require(len(keys) == len(set(keys)), "unique panel rows")
    layers = sorted({row.layer for row in ordered})
    require(len(layers) >= 10, "at least ten layers")
    slots_reference = None
    shape_reference = None
    for layer in layers:
        subset = [row for row in ordered if row.layer == layer]
        slots = sorted({row.slot for row in subset})
        if slots_reference is None:
            slots_reference = slots
        require(slots == slots_reference, "identical expert slots")
        for slot in slots:
            triplet = [row for row in subset if row.slot == slot]
            require([row.role for row in triplet] == list(ROLE_ORDER),
                    "canonical role order per expert")
            shapes = {(row.rows, row.cols) for row in triplet}
            require(len(shapes) == 1, "panel SwiGLU shape equality")
            shape = next(iter(shapes))
            if shape_reference is None:
                shape_reference = shape
            require(shape == shape_reference, "one shape cohort")
    return ordered


def panel_record(rows: Iterable[PanelRow], *, test_layers: int = 5,
                 validation_slots: int | None = None) -> dict[str, Any]:
    panel = canonical_panel(rows)
    layers = sorted({row.layer for row in panel},
                    key=lambda value: (_public_hash("layer", value), value))
    slots = sorted({row.slot for row in panel},
                   key=lambda value: (_public_hash("slot", value), value))
    require(5 <= test_layers < len(layers), "whole test-layer count")
    if validation_slots is None:
        validation_slots = max(1, (len(slots) + 3) // 4)
    require(1 <= validation_slots < len(slots), "validation-slot count")
    test_set = set(layers[:test_layers])
    validation_set = set(slots[:validation_slots])
    rows_record = []
    partition_counts = {"train": 0, "validation": 0, "test": 0}
    for ordinal, row in enumerate(panel):
        if row.layer in test_set:
            partition = "test"
        elif row.slot in validation_set:
            partition = "validation"
        else:
            partition = "train"
        partition_counts[partition] += 1
        item = asdict(row)
        item.update({"component_ordinal": ordinal, "partition": partition})
        rows_record.append(item)
    require(all(partition_counts.values()), "nonempty partitions")
    base = {
        "schema": ADAPTER_SCHEMA + "-panel-v1",
        "rows": rows_record, "test_layers": sorted(test_set),
        "validation_slots": sorted(validation_set),
        "partition_component_counts": partition_counts,
        "split_uses_source_hash": False,
        "control_ordinal_is_canonical_panel_ordinal": True,
    }
    base["panel_sha256"] = sha256(canonical_json(base))
    return base


def control_ordinal(panel: Mapping[str, Any], layer: str, slot: str,
                    role: str) -> int:
    matches = [row for row in panel["rows"]
               if (row["layer"], row["slot"], row["role"]) ==
               (layer, slot, role)]
    require(len(matches) == 1, "canonical control row")
    return int(matches[0]["component_ordinal"])


def selection_receipt(panel: Mapping[str, Any],
                      metrics: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    """Select only from complete train/validation aggregate metrics."""
    require(set(metrics) == {"train", "validation"},
            "selection receives no test metrics")
    config_ids = {config.config_id for config in FROZEN_CONFIGS}
    for partition in ("train", "validation"):
        require(set(metrics[partition]) == config_ids, "complete selection grid")
        for row in metrics[partition].values():
            require(set(row) == {"physical_bits", "weights", "weighted_sse",
                                 "source_energy", "expert_count"},
                    "selection metric schema")
            require(all(math.isfinite(float(value)) and float(value) > 0
                        for value in row.values()), "positive selection metrics")
    def score(config_id: str) -> tuple[Any, ...]:
        row = metrics["validation"][config_id]
        rate = float(row["physical_bits"]) / float(row["weights"])
        distortion = float(row["weighted_sse"]) / float(row["source_energy"])
        f_value = distortion * 2.0 ** (2.0 * rate)
        return (not (RATE_MIN <= rate <= RATE_MAX), f_value, rate, config_id)
    selected = min(config_ids, key=score)
    metrics_record = json.loads(canonical_json(metrics).decode("ascii"))
    receipt = {
        "schema": ADAPTER_SCHEMA + "-selection-receipt-v1",
        "panel_sha256": panel["panel_sha256"],
        "source_hashes_bound_before_test": [row["source_sha256"] for row in panel["rows"]],
        "canonical_control_ordinals": [row["component_ordinal"] for row in panel["rows"]],
        "frozen_grid": frozen_grid_record(),
        "train_validation_metrics": metrics_record,
        "train_validation_metrics_sha256": sha256(canonical_json(metrics_record)),
        "selected_config_id": selected,
        "selection_key": list(score(selected)),
        "test_metrics_opened_or_accepted": False,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    return receipt


def authorize_test(panel: Mapping[str, Any], receipt: Mapping[str, Any]) -> PilotConfig:
    require(receipt.get("panel_sha256") == panel.get("panel_sha256"),
            "test panel binding")
    check = dict(receipt)
    claimed = check.pop("receipt_sha256", None)
    require(claimed == sha256(canonical_json(check)), "selection receipt seal")
    require(receipt.get("test_metrics_opened_or_accepted") is False,
            "selection excluded test")
    require(receipt.get("frozen_grid") == frozen_grid_record(),
            "selection frozen grid")
    expected_hashes = [row["source_sha256"] for row in panel["rows"]]
    expected_ordinals = list(range(len(panel["rows"])))
    require(receipt.get("source_hashes_bound_before_test") == expected_hashes and
            receipt.get("canonical_control_ordinals") == expected_ordinals,
            "test source and ordinal binding")
    matches = [config for config in FROZEN_CONFIGS
               if config.config_id == receipt.get("selected_config_id")]
    require(len(matches) == 1, "selected config")
    return matches[0]


def moment_matched_gaussian(np: Any, source: Any, *, block_size: int,
                            seed: int, component_ordinal: int) -> tuple[Any, dict[str, Any]]:
    values = np.asarray(source, dtype=np.float64).reshape(-1)
    require(seed in CONTROL_SEEDS and component_ordinal >= 0 and
            values.size and values.size % block_size == 0,
            "matched control inputs")
    generated = np.empty_like(values)
    moments = []
    for block_index, start in enumerate(range(0, values.size, block_size)):
        block = values[start:start + block_size]
        mean = float(np.mean(block, dtype=np.float64))
        centered = block - mean
        energy = float(np.sum(centered * centered, dtype=np.float64))
        material = (CONTROL_DOMAIN + struct.pack(">QQQ", seed, component_ordinal,
                                                 block_index))
        counter = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        rng = np.random.Generator(np.random.PCG64(counter))
        z = rng.standard_normal(block_size, dtype=np.float64)
        z -= float(np.mean(z, dtype=np.float64))
        z_energy = float(np.sum(z * z, dtype=np.float64))
        require(z_energy > 0.0, "control nondegenerate")
        control = (np.full(block_size, mean, dtype=np.float64) if energy == 0.0
                   else mean + z * math.sqrt(energy / z_energy))
        control -= float(np.mean(control, dtype=np.float64)) - mean
        if energy:
            residual = control - mean
            control = mean + residual * math.sqrt(
                energy / float(np.sum(residual * residual, dtype=np.float64)))
        generated[start:start + block_size] = control
        observed_mean = float(np.mean(control, dtype=np.float64))
        observed_energy = float(np.sum((control - observed_mean) ** 2,
                                       dtype=np.float64))
        moments.append({"block": block_index, "source_mean_hex": mean.hex(),
                        "source_centered_sse_hex": energy.hex(),
                        "control_mean_hex": observed_mean.hex(),
                        "control_centered_sse_hex": observed_energy.hex()})
    return generated, {
        "seed": seed, "component_ordinal": component_ordinal,
        "source_sha256": sha256(_to_numpy(np, values).tobytes(order="C")),
        "control_sha256": sha256(_to_numpy(np, generated).tobytes(order="C")),
        "block_moments": moments, "prebuilt_labels_accepted": False,
    }


def rerun_matched_controls(np: Any, xp: Any, core: Any,
                           roles: Mapping[str, tuple[Any, Any]], *,
                           layer: str, slot: str, rows: int, cols: int,
                           config: PilotConfig, panel: Mapping[str, Any],
                           live: bool = False) -> dict[str, Any]:
    """Regenerate moments and rerun the complete capped family bank for all seeds."""
    require(set(roles) == set(ROLE_ORDER), "control expert roles")
    results = []
    for seed in CONTROL_SEEDS:
        control_roles = {}
        receipts = {}
        for role in ROLE_ORDER:
            ordinal = control_ordinal(panel, layer, slot, role)
            control, receipt = moment_matched_gaussian(
                np, _to_numpy(xp, roles[role][0]), block_size=config.block_size,
                seed=seed, component_ordinal=ordinal)
            control_roles[role] = (xp.asarray(control, dtype=xp.float64),
                                   roles[role][1])
            receipts[role] = receipt
        encoded = encode_expert(xp, core, control_roles, rows=rows, cols=cols,
                                config=config, live=live)
        results.append({"seed": seed, "score": encoded["score"],
                        "generation": receipts,
                        "packet_sha256": sha256(encoded["packet"])})
    require([row["seed"] for row in results] == list(CONTROL_SEEDS),
            "complete matched-control seeds")
    return {"schema": ADAPTER_SCHEMA + "-matched-controls-v1",
            "full_capped_pipeline_rerun_including_presearch_decisions": True,
            "controls_can_create_absolute_source_pass": False,
            "results": results}
