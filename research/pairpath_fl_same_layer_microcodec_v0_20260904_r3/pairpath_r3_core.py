"""Source-only PAIRPATH-P2 r3 repair layer.

This module cryptographically pins and loads the sealed r2 implementation, then
replaces only the three surfaces blocked by its independent hostile audit:

* one global Up/Down distortion-per-bit multiplier is used by both finite roles;
* the heuristic joint oracle includes and certifies the independent solution;
* literal packet decoding validates and causally replays the tree descriptor.

There is deliberately no payload locator, GPU entry point, or execution grant.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Mapping, Sequence

import numpy as np


R2_CORE_SHA256 = "2c99a31aef669cabbb67137061233640b013e8c50a5132ddbcc9ffec2c239034"
R2_CORE_PATH = (Path(__file__).resolve().parent.parent /
                "pairpath_fl_same_layer_microcodec_v0_20260903_r2" /
                "pairpath_r2_core.py")
_r2_bytes = R2_CORE_PATH.read_bytes()
if hashlib.sha256(_r2_bytes).hexdigest() != R2_CORE_SHA256:
    raise RuntimeError("pinned r2 core dependency mismatch")
exec(compile(_r2_bytes, str(R2_CORE_PATH), "exec"), globals(), globals())


R3_REPAIR_SCHEMA = "pairpath_p2_r3_repair_certificate_v1"
_r2_encode_plan = _encode_plan
_r2_parse_packet = _parse_packet


def global_updown_bit_weight(values: np.ndarray, lagrange: Fraction) -> float:
    """Return the one exact multiplier shared by finite Up and Down fitting."""
    x = _validate_values(values, (2,))
    require(isinstance(lagrange, Fraction) and lagrange in LAMBDA_GRID,
            "frozen lambda")
    optimized = x[:, OPTIMIZED_ROLES]
    energy = float(np.sum(optimized * optimized, dtype=np.float64))
    return (float(lagrange) * max(energy, np.finfo(np.float64).tiny) /
            optimized.size)


def choose_pair_labels(values: np.ndarray, scale_bits: np.ndarray, lagrange: Fraction,
                       flexible: bool, *, global_bit_weight: float | None = None) -> dict:
    """Fit one role, refusing the role-local multiplier that invalidated r2."""
    x = np.asarray(values, dtype=np.float64)
    s = np.asarray(scale_bits)
    require(x.ndim == 2 and x.shape[0] == 2 and s.shape ==
            (2, x.shape[1] // BLOCK_VALUES) and s.dtype == np.uint16,
            "pair role geometry")
    require(isinstance(lagrange, Fraction) and lagrange in LAMBDA_GRID,
            "frozen lambda")
    require(global_bit_weight is not None and math.isfinite(global_bit_weight) and
            global_bit_weight >= 0, "explicit global Up/Down bit weight required")
    bit_weight = float(global_bit_weight)
    folds = fold_ids(x.shape[1])
    levels = np.stack([levels_per_coordinate(s[e], x.shape[1]) for e in range(2)])
    nearest = np.stack([nearest_labels(x[e], s[e]) for e in range(2)])
    models = []
    states = np.empty(x.shape[1], np.uint8)
    labels = np.empty(x.shape, np.uint8)
    for fold in range(FOLD_COUNT):
        train, held = folds != fold, folds == fold
        model = _fit_pair_fold(x, levels, nearest, train, flexible, bit_weight)
        state, q = _apply_pair_fold(x, levels, nearest, held, model, flexible,
                                    bit_weight)
        models.append(model)
        states[held], labels[:, held] = state, q
    require(bool(np.all(states < STATE_COUNT)) and bool(np.all(labels < ALPHABET)),
            "pair output range")
    return {"models": models, "states": states, "labels": labels,
            "nearest": nearest, "global_bit_weight": bit_weight,
            "global_bit_weight_hex": bit_weight.hex()}


def _make_plan(values: np.ndarray, candidate: str, lagrange: Fraction) -> dict:
    """Make a finite plan with a single globally normalized Up/Down multiplier."""
    x = _validate_values(values, (2,))
    require(candidate in CANDIDATES and lagrange in LAMBDA_GRID, "candidate request")
    scales = estimate_scale_bits(x)
    independent_models, independent_labels = _independent_models_and_labels(x, scales)
    labels = independent_labels.copy()
    pair_models, states = {}, {}
    bit_weight = global_updown_bit_weight(x, lagrange)
    role_weights = {}
    if candidate != "independent_fixed":
        for role in OPTIMIZED_ROLES:
            result = choose_pair_labels(
                x[:, role], scales[:, role], lagrange,
                candidate == "pair_k2_flexible", global_bit_weight=bit_weight)
            pair_models[str(role)] = result["models"]
            states[str(role)] = result["states"]
            labels[:, role] = result["labels"]
            role_weights[str(role)] = result["global_bit_weight_hex"]
    certificate = {
        "schema": R3_REPAIR_SCHEMA,
        "normalization": "one global Up/Down source energy divided by all Up/Down weights",
        "lambda": str(lagrange),
        "global_bit_weight_hex": bit_weight.hex(),
        "optimized_role_bit_weight_hex": role_weights,
    }
    if candidate != "independent_fixed":
        require(len(set(role_weights.values())) == 1 and
                next(iter(role_weights.values())) == bit_weight.hex(),
                "finite role multiplier divergence")
    return {"candidate": candidate, "lambda": str(lagrange), "scales": scales,
            "labels": labels, "independent_models": independent_models,
            "pair_models": pair_models, "states": states,
            "r3_encoder_certificate": certificate}


def _packet_parts(packet: bytes) -> tuple[dict, bytes, list[bytes]]:
    """Extract packet members without invoking the repaired parser."""
    require(isinstance(packet, bytes) and len(packet) >= 12 and packet[:8] == MAGIC,
            "packet magic")
    header_len = struct.unpack("<I", packet[8:12])[0]
    raw_header = packet[12:12 + header_len]
    header = json.loads(raw_header)
    require(canonical_json(header) == raw_header, "canonical header")
    common_size = int(header["common_pages"]) * PAGE_BYTES
    common_start = 12 + header_len
    common_end = common_start + int(header["common_payload_bytes"])
    require(common_end <= common_size, "common allocation")
    common_payload = packet[common_start:common_end]
    private_payloads = []
    offset = common_size
    for raw_size, pages in zip(header["private_payload_bytes"], header["private_pages"]):
        segment = packet[offset:offset + int(pages) * PAGE_BYTES]
        private_payloads.append(segment[:int(raw_size)])
        offset += int(pages) * PAGE_BYTES
    require(offset == len(packet), "packet extent")
    return header, common_payload, private_payloads


def _pack_parts(header: Mapping, common_payload: bytes,
                private_payloads: Sequence[bytes]) -> bytes:
    """Canonical packet rebuild used both by the encoder and hostile KATs."""
    base = dict(header)
    base["common_payload_bytes"] = len(common_payload)
    base["private_payload_bytes"] = [len(v) for v in private_payloads]
    base["common_pages"] = 0
    base["private_pages"] = [0, 0]
    current = dict(base)
    for _ in range(16):
        raw_header = canonical_json(current)
        common_pages = ceil_div(12 + len(raw_header) + len(common_payload), PAGE_BYTES)
        private_pages = [ceil_div(len(v), PAGE_BYTES) for v in private_payloads]
        total_weights = math.prod(current["shape"])
        min_pages = ceil_div(ceil_div(RATE_MIN.numerator * total_weights,
                                     RATE_MIN.denominator * 8), PAGE_BYTES)
        while common_pages + sum(private_pages) < min_pages:
            target = 0 if private_pages[0] <= private_pages[1] else 1
            private_pages[target] += 1
        updated = dict(base)
        updated["common_pages"] = common_pages
        updated["private_pages"] = private_pages
        if updated == current:
            break
        current = updated
    else:
        raise CodecError("r3 packet header did not converge")
    raw_header = canonical_json(current)
    common_raw = MAGIC + struct.pack("<I", len(raw_header)) + raw_header + common_payload
    segments = [common_raw + bytes(current["common_pages"] * PAGE_BYTES - len(common_raw))]
    for payload, pages in zip(private_payloads, current["private_pages"]):
        segments.append(payload + bytes(int(pages) * PAGE_BYTES - len(payload)))
    packet = b"".join(segments)
    rate = Fraction(len(packet) * 8, math.prod(current["shape"]))
    require(RATE_MIN <= rate <= RATE_MAX, "literal packet outside rate interval")
    return packet


def _encode_plan(values: np.ndarray, plan: Mapping) -> bytes:
    packet = _r2_encode_plan(values, plan)
    header, common_payload, private_payloads = _packet_parts(packet)
    header["r3_encoder_certificate"] = plan["r3_encoder_certificate"]
    return _pack_parts(header, common_payload, private_payloads)


def _tree_as_json(tree):
    return int(tree) if isinstance(tree, int) else [_tree_as_json(tree[0]),
                                                    _tree_as_json(tree[1])]


def _validate_and_replay_tree_descriptor(header: Mapping) -> dict:
    """Fail closed unless every redundant descriptor field replays exactly."""
    descriptor = header.get("tree_descriptor")
    require(isinstance(descriptor, dict) and set(descriptor) ==
            {"packed", "bits", "pairs", "merge_ranks", "materialized"},
            "tree descriptor schema")
    expert_count = int(header["shape"][0])
    packed, bits = descriptor["packed"], descriptor["bits"]
    require(type(packed) is int and type(bits) is int, "tree descriptor integers")
    decoded = decode_tree_descriptor(packed, expert_count)
    replay_packed, replay_bits = encode_tree_descriptor(
        expert_count, decoded["pairs"], decoded["merge_ranks"])
    expected_pairs = [list(pair) for pair in decoded["pairs"]]
    expected_merges = list(decoded["merge_ranks"])
    expected_tree = _tree_as_json(decoded["tree"])
    require(bits == tree_descriptor_bits(expert_count) and
            (replay_packed, replay_bits) == (packed, bits),
            "tree descriptor causal replay")
    require(descriptor["pairs"] == expected_pairs and
            descriptor["merge_ranks"] == expected_merges and
            descriptor["materialized"] == expected_tree,
            "tree descriptor redundant fields")
    return decoded


def _parse_packet(packet: bytes) -> tuple[dict, bytes, list[bytes]]:
    parsed = _r2_parse_packet(packet)
    header = parsed[0]
    _validate_and_replay_tree_descriptor(header)
    certificate = header.get("r3_encoder_certificate")
    require(isinstance(certificate, dict) and
            certificate.get("schema") == R3_REPAIR_SCHEMA,
            "missing r3 encoder certificate")
    role_weights = certificate.get("optimized_role_bit_weight_hex")
    require(isinstance(role_weights, dict), "role multiplier certificate")
    if header["candidate"] == "independent_fixed":
        require(role_weights == {}, "independent role multiplier certificate")
    else:
        expected = certificate.get("global_bit_weight_hex")
        require(set(role_weights) == {str(v) for v in OPTIMIZED_ROLES} and
                all(value == expected for value in role_weights.values()),
                "finite role multiplier divergence")
    return parsed


def _empirical_role_score(values: np.ndarray, levels: np.ndarray, labels: np.ndarray,
                          bit_weight: float, joint: bool) -> tuple[float, float, float]:
    reconstruction = np.take_along_axis(levels, labels[:, :, None], axis=2)[:, :, 0]
    sse = float(np.sum((values - reconstruction) ** 2, dtype=np.float64))
    if joint:
        index = labels[0].astype(np.int16) * ALPHABET + labels[1]
        rate = _entropy_bits(np.bincount(index, minlength=ALPHABET * ALPHABET)) / 2.0
    else:
        rate = sum(_entropy_bits(np.bincount(labels[e], minlength=ALPHABET))
                   for e in range(2)) / 2.0
    objective = sse + bit_weight * rate * values.size
    return objective, sse, rate


def _ideal_flexible_role_certified(values: np.ndarray, levels: np.ndarray,
                                    bit_weight: float, joint: bool, *,
                                    required_candidate: np.ndarray | None = None) -> dict:
    """Heuristic search with an exact dominance certificate over a supplied candidate."""
    require(values.shape[0] == 2 and levels.shape == values.shape + (ALPHABET,),
            "ideal role geometry")
    require(math.isfinite(bit_weight) and bit_weight >= 0, "ideal bit weight")
    starts = _ideal_initializations(values, levels)
    if required_candidate is not None:
        candidate = np.asarray(required_candidate, dtype=np.uint8)
        require(candidate.shape == values.shape and bool(np.all(candidate < ALPHABET)),
                "required candidate geometry")
        starts = [candidate.copy()] + starts
    best = None
    for start_index, start in enumerate(starts):
        q = start.copy()
        visited = set()
        for iteration in range(MAX_ALTERNATIONS + 1):
            objective, sse, rate = _empirical_role_score(
                values, levels, q, bit_weight, joint)
            key = (objective, sse, rate, start_index, iteration, q.tobytes())
            if best is None or key < best[0]:
                best = (key, q.copy(), sse, rate)
            packed = q.tobytes()
            if iteration == MAX_ALTERNATIONS or packed in visited:
                break
            visited.add(packed)
            if joint:
                index = q[0].astype(np.int16) * ALPHABET + q[1]
                counts = np.bincount(index, minlength=ALPHABET * ALPHABET).astype(np.float64)
                length = -np.log2((counts + 0.5) / (index.size + 0.5 * counts.size))
                costs = np.empty((values.shape[1], ALPHABET * ALPHABET), np.float64)
                for a0 in range(ALPHABET):
                    for a1 in range(ALPHABET):
                        k = a0 * ALPHABET + a1
                        costs[:, k] = ((values[0] - levels[0, :, a0]) ** 2 +
                                       (values[1] - levels[1, :, a1]) ** 2 +
                                       bit_weight * length[k])
                selected = np.argmin(costs, axis=1)
                q = np.stack((selected // ALPHABET, selected % ALPHABET)).astype(np.uint8)
            else:
                new_q = np.empty_like(q)
                for expert in range(2):
                    counts = np.bincount(q[expert], minlength=ALPHABET).astype(np.float64)
                    length = -np.log2((counts + 0.5) /
                                      (q.shape[1] + 0.5 * ALPHABET))
                    costs = ((values[expert, :, None] - levels[expert]) ** 2 +
                             bit_weight * length[None, :])
                    new_q[expert] = np.argmin(costs, axis=1).astype(np.uint8)
                q = new_q
    require(best is not None, "ideal multistart result")
    candidate_score = None
    if required_candidate is not None:
        candidate_score = _empirical_role_score(
            values, levels, np.asarray(required_candidate), bit_weight, joint)
        require(best[0][0] <= candidate_score[0] + 1e-12 * max(1.0, abs(candidate_score[0])),
                "joint solver failed required-candidate dominance")
    return {
        "labels": best[1], "objective": best[0][0], "sse": best[2], "rate_bpw": best[3],
        "required_candidate_objective": None if candidate_score is None else candidate_score[0],
        "dominates_required_candidate": candidate_score is None or best[0][0] <=
            candidate_score[0] + 1e-12 * max(1.0, abs(candidate_score[0])),
        "global_optimality_proven": False,
    }


def _ideal_flexible_role(values: np.ndarray, levels: np.ndarray, bit_weight: float,
                         joint: bool) -> tuple[np.ndarray, float, float]:
    result = _ideal_flexible_role_certified(values, levels, bit_weight, joint)
    return result["labels"], result["sse"], result["rate_bpw"]


def optimistic_single_letter_joint_gate(values: np.ndarray,
                                        lambda_grid: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    """Dominance-certified heuristic RD census; it has no hard-kill authority."""
    x = _validate_values(values, (2,))
    scales = estimate_scale_bits(x)
    grid = (Fraction(0, 1),) + tuple(lambda_grid)
    require(all(v == 0 or v in LAMBDA_GRID for v in grid), "oracle lambda grid")
    independent_points, pair_points, certificates = [], [], []
    energy = float(np.sum(x[:, OPTIMIZED_ROLES] ** 2, dtype=np.float64))
    for lagrange in grid:
        bit_weight = (float(lagrange) * max(energy, np.finfo(np.float64).tiny) /
                      x[:, OPTIMIZED_ROLES].size)
        independent_sse = pair_sse = 0.0
        independent_rate_sum = pair_rate_sum = 0.0
        for role in OPTIMIZED_ROLES:
            levels = np.stack([levels_per_coordinate(scales[e, role], x.shape[2])
                               for e in range(2)])
            independent = _ideal_flexible_role_certified(
                x[:, role], levels, bit_weight, False)
            pair = _ideal_flexible_role_certified(
                x[:, role], levels, bit_weight, True,
                required_candidate=independent["labels"])
            independent_sse += independent["sse"]
            pair_sse += pair["sse"]
            independent_rate_sum += independent["rate_bpw"]
            pair_rate_sum += pair["rate_bpw"]
            certificates.append({
                "lambda": str(lagrange), "role": ROLES[role],
                "bit_weight_hex": bit_weight.hex(),
                "joint_objective": pair["objective"],
                "independent_labels_under_joint_objective":
                    pair["required_candidate_objective"],
                "dominates_independent_candidate": pair["dominates_required_candidate"],
                "global_optimality_proven": False,
            })
        independent_points.append((independent_rate_sum / len(OPTIMIZED_ROLES),
                                   independent_sse / energy))
        pair_points.append((pair_rate_sum / len(OPTIMIZED_ROLES), pair_sse / energy))
    require(all(row["dominates_independent_candidate"] for row in certificates),
            "oracle dominance certificate")
    independent_hull, pair_hull = _pareto_rd(independent_points), _pareto_rd(pair_points)
    lo = max(independent_hull[0][0], pair_hull[0][0])
    hi = min(independent_hull[-1][0], pair_hull[-1][0])
    rates = sorted({lo, hi} | {r for r, _ in independent_hull if lo <= r <= hi} |
                   {r for r, _ in pair_hull if lo <= r <= hi})
    equal_rate = []
    if lo <= hi:
        for rate in rates:
            di = _interp_distortion(independent_hull, rate)
            dp = _interp_distortion(pair_hull, rate)
            equal_rate.append({"rate_bpw": rate, "D_ind": di, "D_pair": dp,
                               "G_eq_bpw": 0.5 * math.log2(di / dp)})
    dlo = max(independent_hull[-1][1], pair_hull[-1][1])
    dhi = min(independent_hull[0][1], pair_hull[0][1])
    distortions = sorted({dlo, dhi} |
                         {d for _, d in independent_hull if dlo <= d <= dhi} |
                         {d for _, d in pair_hull if dlo <= d <= dhi}, reverse=True)
    equal_mse = []
    if dlo <= dhi:
        for distortion in distortions:
            ri = _interp_rate(independent_hull, distortion)
            rp = _interp_rate(pair_hull, distortion)
            equal_mse.append({"relative_D": distortion, "R_ind_bpw": ri,
                              "R_pair_bpw": rp, "G_eq_bpw": ri - rp})
    gains = [row["G_eq_bpw"] for row in equal_rate + equal_mse]
    best_gain = max(gains) if gains else -math.inf
    if best_gain >= ORACLE_ENGINEERING_MARGIN_BPW:
        status = "SURVIVE_HEURISTIC_WITH_PHYSICAL_MARGIN"
    elif best_gain >= REQUIRED_UPDOWN_GAIN_BPW:
        status = "SURVIVE_HEURISTIC_STANDALONE_THRESHOLD"
    elif best_gain >= ORACLE_EARLY_KILL_BPW:
        status = "INTERESTING_HEURISTIC_BUT_INSUFFICIENT_STANDALONE"
    else:
        status = "HEURISTIC_BELOW_GATE_NONAUTHORITATIVE"
    return {
        "schema": "pairpath_p2_r3_dominance_certified_heuristic_gate_v1",
        "status": status,
        "claim": "heuristic census only; never hard-kill because global optimality is unproven",
        "hard_kill_authority": False,
        "global_optimality_proven": False,
        "independent_candidate_dominance_certificates": certificates,
        "fixed_assignment_mi": fixed_assignment_mi_ceiling(x),
        "independent_hull": independent_hull, "pair_hull": pair_hull,
        "equal_rate": equal_rate, "equal_mse": equal_mse,
        "best_G_eq_UD_bpw": best_gain, "early_kill_bpw": ORACLE_EARLY_KILL_BPW,
        "standalone_required_bpw": REQUIRED_UPDOWN_GAIN_BPW,
        "physical_engineering_margin_bpw": ORACLE_ENGINEERING_MARGIN_BPW,
    }


def hard_kill_contract() -> dict:
    contract = globals().get("hard_kill_contract_r2")
    return {
        "r3_oracle": "dominance-certified heuristic has no hard-kill authority",
        "finite": "one independently decoded packet must have F<=0.8, 2.15<=R<=2.5, max read <2",
        "promotion": "requires controls and at least two held-out layers; source-only package cannot promote",
    }
