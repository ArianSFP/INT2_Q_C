#!/usr/bin/env python3
"""Independent, payload-free hostile audit for label-copula stage 0.

This script is intentionally outside the producer package.  It reads only the
sealed source package named on the command line and generates only synthetic
bit streams and temporary lifecycle fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import struct
import sys
import tempfile
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "e1bc2873f204b1db5fefa666d0daf6ddebae38bd3f1add3ce45f3bc0538aae14"
Q_TOTAL = 1 << 16
FULL = 1 << 32
HALF = 1 << 31
QUARTER = 1 << 30
THREE_QUARTERS = 3 << 30


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            value.update(block)
    return value.hexdigest()


def load_common(package: Path):
    path = package / "label_copula_common.py"
    spec = importlib.util.spec_from_file_location("audited_label_copula_common", path)
    check(spec is not None and spec.loader is not None, "common import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_context(role: int, plane: int, position: int, reset: int) -> int:
    within = position % reset
    remaining = reset - 1 - within
    boundary = remaining if remaining < 8 else 8
    phase = within & 7
    return (((role * 2 + plane) * 8 + phase) * 9) + boundary


def independent_next(candidate, state: int, symbol: int, role: int, plane: int, position: int) -> int:
    topology = candidate.topology
    chi = candidate.chi
    if topology == "factorized":
        return 0
    mask = chi - 1
    within = position % candidate.reset
    if topology == "suffix":
        return ((state << 1) | symbol) & mask
    if topology == "parity_sketch":
        bits = chi.bit_length() - 1
        if symbol and within < candidate.reset - bits:
            return state ^ (1 << ((within + 3 * role + 5 * plane) % bits))
        return state
    if topology == "modular":
        half = max(1, chi // 2)
        step = (2 * ((within + 3 * role + 5 * plane) % half) + 1) & mask
        return (state + symbol * step) & mask
    if topology == "rolling":
        return (state * (5 if chi >= 8 else 3) + symbol) & mask
    if topology == "regime":
        half = chi // 2
        last, age = divmod(state, half)
        if symbol == last:
            age = min(half - 1, age + 1)
        else:
            last, age = symbol, 0
        return last * half + age
    raise AssertionError(f"unknown topology {topology}")


class IndependentBitReader:
    def __init__(self, payload: bytes, meaningful: int):
        self.payload = payload
        self.meaningful = meaningful
        self.position = 0

    def read(self) -> int:
        position = self.position
        self.position += 1
        if position >= self.meaningful:
            return 0
        return (self.payload[position >> 3] >> (7 - (position & 7))) & 1


class IndependentDecoder:
    def __init__(self, payload: bytes, meaningful: int):
        self.reader = IndependentBitReader(payload, meaningful)
        self.low = 0
        self.high = FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self.reader.read()) & (FULL - 1)

    def read(self, freq1: int) -> int:
        f0 = Q_TOTAL - freq1
        width = self.high - self.low + 1
        split = self.low + width * f0 // Q_TOTAL - 1
        if self.code <= split:
            symbol = 0
            self.high = split
        else:
            symbol = 1
            self.low = split + 1
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & (FULL - 1)
            self.high = ((self.high << 1) & (FULL - 1)) | 1
            self.code = ((self.code << 1) & (FULL - 1)) | self.reader.read()
        return symbol


def independent_decode(packet: bytes, roles: tuple[int, ...], planes: tuple[int, ...], payload: bytes, meaningful: int):
    header = struct.unpack("<8sHHHHIIII", packet[:32])
    magic, version, code, chi, reserved, reset, contexts, total, count = header
    check(magic == b"LCWFA0\0\0" and version == 0 and reserved == 0, "model header identity")
    check(contexts == 432 and total == Q_TOTAL and count == contexts * chi, "model header geometry")
    table = struct.unpack(f"<{count}H", packet[256:])
    topology = "factorized" if code == 0 else ("suffix", "parity_sketch", "modular", "rolling", "regime")[code - 1]
    candidate = type("IndependentCandidate", (), {"topology": topology, "chi": chi, "reset": reset})()
    decoder = IndependentDecoder(payload, meaningful)
    state = 0
    output = []
    for position, (role, plane) in enumerate(zip(roles, planes, strict=True)):
        if position % reset == 0:
            state = 0
        context = independent_context(role, plane, position, reset)
        output.append(decoder.read(table[context * chi + state]))
        state = independent_next(candidate, state, output[-1], role, plane, position)
    return tuple(output)


def tiny_stream(common, layer: str, expert: str, salt: int, symbols: int = 4096):
    rng = random.Random(99173 + salt)
    bits = tuple(rng.getrandbits(1) for _ in range(symbols))
    roles = tuple((index // 2) % 3 for index in range(symbols))
    planes = tuple(index & 1 for index in range(symbols))
    return common.SymbolStream(layer, expert, bits, roles, planes, symbols // 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.absolute()

    manifest_path = package / "SOURCE_MANIFEST.json"
    manifest_sha = digest(manifest_path)
    check(manifest_sha == EXPECTED_MANIFEST_SHA256, "external manifest binding")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    expected_names = {row["name"] for row in manifest["files"]} | {"SOURCE_MANIFEST.json"}
    observed_names = {entry.name for entry in os.scandir(package)}
    check(observed_names == expected_names, "external package closure")
    for row in manifest["files"]:
        member = package / row["name"]
        check(member.is_file() and not member.is_symlink(), f"regular source member {row['name']}")
        check(member.stat().st_size == row["bytes"], f"source bytes {row['name']}")
        check(digest(member) == row["sha256"], f"source digest {row['name']}")

    common = load_common(package)
    findings = []
    passes = []

    # Canonical semantic orientation, including non-square and singleton axes.
    gate = [[100 * j + k + 0.1 for k in range(5)] for j in range(3)]
    up = [[100 * j + k + 0.2 for k in range(5)] for j in range(3)]
    down = [[100 * j + k + 0.3 for j in range(3)] for k in range(5)]
    values, roles = common.canonical_swiglu_values(gate, up, down)
    expected_values = tuple(value for j in range(3) for k in range(5) for value in (gate[j][k], up[j][k], down[k][j]))
    check(values == expected_values, "canonical Gate/Up/Down order")
    check(roles == tuple((0, 1, 2) * 15), "canonical semantic roles")
    singleton, _ = common.canonical_swiglu_values([[1.0]], [[2.0]], [[3.0]])
    check(singleton == (1.0, 2.0, 3.0), "singleton canonical shape")
    passes.append("canonical arbitrary-shape semantic orientation")

    # Frozen Gaussian-Lloyd4 threshold and deterministic raw-label reference.
    threshold = 0.981598821873
    check(common.lloyd4_label(-threshold - 1e-12, 1.0) == 0, "lower outer Lloyd cell")
    check(common.lloyd4_label(-threshold, 1.0) == 1, "lower Lloyd boundary")
    check(common.lloyd4_label(0.0, 1.0) == 2, "zero Lloyd boundary")
    check(common.lloyd4_label(threshold, 1.0) == 2, "upper Lloyd boundary")
    check(common.lloyd4_label(threshold + 1e-12, 1.0) == 3, "upper outer Lloyd cell")
    passes.append("fixed RMS/Gaussian-Lloyd4 labeler boundary")

    # Exhaustive recurrence cross-check over every frozen cell and key edge.
    for candidate in common.candidate_bank() + common.factorized_bank():
        states = range(candidate.chi)
        positions = {0, 1, 7, 8, candidate.reset - 7, candidate.reset - 1, candidate.reset, candidate.reset + 1}
        for state in states:
            for symbol in (0, 1):
                for role in range(3):
                    for plane in range(2):
                        for position in positions:
                            observed = common.next_state(candidate, state, symbol, role, plane, position)
                            expected = independent_next(candidate, state, symbol, role, plane, position)
                            check(observed == expected and 0 <= observed < candidate.chi, "unifilar state law")
    passes.append("240-cell exact-integer unifilar topology bank")

    # Cross-implementation model-packet and arithmetic roundtrip, including
    # probability extremes and every topology family.
    training = tuple(tiny_stream(common, "train", f"slot-{index}", index) for index in range(4))
    target = tiny_stream(common, "test", "slot-x", 19, symbols=8192)
    for topology in ("factorized", "suffix", "parity_sketch", "modular", "rolling", "regime"):
        candidate = common.Candidate(topology, 1 if topology == "factorized" else 8, 64)
        model = common.fit_model(training, candidate)
        packet = model.serialize()
        check(packet[32:256] == bytes(224), "zero model-header padding")
        restored = common.QuantizedModel.deserialize(packet)
        payload, meaningful = common.encode_stream(restored, target)
        decoded = independent_decode(packet, target.roles, target.planes, payload, meaningful)
        check(decoded == target.symbols, f"independent arithmetic decode {topology}")
    for bit, frequency in ((0, 1), (0, 65535), (1, 1), (1, 65535)):
        candidate = common.Candidate("factorized", 1, 32)
        model = common.QuantizedModel(candidate, tuple(frequency for _ in range(432)))
        stream = common.SymbolStream("L", "E", tuple([bit] * 8192), tuple([0] * 8192), tuple([0] * 8192), 4096)
        packet = model.serialize()
        payload, meaningful = common.encode_stream(model, stream)
        check(independent_decode(packet, stream.roles, stream.planes, payload, meaningful) == stream.symbols, "extreme arithmetic frequency")
    passes.append("Q0.16 packet plus independent finite arithmetic decoding")

    # Probability fit and coding must ignore reporting/split identity.
    original = tiny_stream(common, "layer-a", "expert-a", 23)
    renamed = common.SymbolStream("different-layer", "different-expert", original.symbols, original.roles, original.planes, original.source_weights)
    candidate = common.Candidate("rolling", 8, 128)
    check(common.fit_model((original,), candidate) == common.fit_model((renamed,), candidate), "identity-free probability fit")
    passes.append("layer/expert strings absent from probability key")

    # Nested regular-grid split isolation.
    grid = tuple(tiny_stream(common, f"layer-{layer}", f"slot-{expert}", 1000 * layer + expert, symbols=512)
                 for layer in range(10) for expert in range(8))
    folds = common.nested_partition(grid)
    check(not ({row.layer_group for row in folds["train"]} & {row.layer_group for row in folds["test"]}), "outer whole-layer isolation")
    check(not ({row.expert_group for row in folds["train"]} & {row.expert_group for row in folds["validation"]}), "inner whole-slot isolation")
    check(len({row.layer_group for row in folds["test"]}) == 2, "frozen 20-percent outer split")
    passes.append("whole-layer outer and whole-slot inner regular-grid split")

    # Literal ledger recomputation.
    model = common.fit_model(training, common.Candidate("suffix", 4, 64))
    encoded = tuple((stream, *common.encode_stream(model, stream)) for stream in training)
    ledger = common.container_ledger(len(model.serialize()), encoded)
    model_stored = math.ceil(len(model.serialize()) / 4096) * 4096
    directory_stored = math.ceil((64 * len(encoded)) / 4096) * 4096
    offset = 4096 + model_stored + directory_stored
    for stream, payload, meaningful in encoded:
        check(len(payload) == math.ceil(meaningful / 8), "literal arithmetic byte padding")
        offset += math.ceil((256 + len(payload)) / 64) * 64
    total = math.ceil(offset / 4096) * 4096
    check(total == ledger["total_physical_bytes"], "independent total physical bytes")
    check(total - offset == ledger["final_container_padding_bytes"], "independent final padding")
    passes.append("model/header/directory/frame/alignment/page/read ledger arithmetic")

    # Threshold and no-control-created-pass contract.
    threshold_expected = -0.5 * math.log2(0.8) - 0.008074080480766676
    check(abs(common.STANDALONE_REQUIRED_SAVING_BPW - threshold_expected) <= 2e-16, "physical target threshold")
    check(not common.matched_control_gate({"net_nonlocal_physical_saving_bpw": threshold_expected + 1.0,
                                           "absolute_source_survival_before_controls": False}), "control cannot rescue lower-bound miss")
    passes.append("0.15288996696 physical diagnostic gate and no-control rescue")

    # Hostile lifecycle probes.  Record rather than abort: these decide status.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "output"
        with common.CompletionLastOutput(output) as writer:
            writer.complete("a" * 64)
            try:
                writer.write_new("AFTER_COMPLETE.json", b"{}\n")
            except Exception:
                post_complete_write = False
            else:
                post_complete_write = True
        if post_complete_write:
            findings.append({
                "id": "LIFECYCLE_COMPLETE_NOT_FINAL",
                "severity": "BLOCK",
                "detail": "CompletionLastOutput permits write_new after COMPLETE.json, contradicting exclusive-last acceptance semantics.",
            })

    # The verifier resolves the package before checking symlink identity.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        link = root / "package-link"
        try:
            link.symlink_to(package, target_is_directory=True)
        except (OSError, NotImplementedError):
            symlink_accepted = None
        else:
            verifier_path = package / "verify_source.py"
            spec = importlib.util.spec_from_file_location("audited_verify_source", verifier_path)
            check(spec is not None and spec.loader is not None, "verifier import spec")
            verifier = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = verifier
            spec.loader.exec_module(verifier)
            try:
                verifier.verify_package(link)
            except Exception:
                symlink_accepted = False
            else:
                symlink_accepted = True
        if symlink_accepted:
            findings.append({
                "id": "SOURCE_PACKAGE_SYMLINK_ACCEPTED",
                "severity": "BLOCK",
                "detail": "verify_package resolves the argument before its symlink check, so a symlinked package is accepted.",
            })

    # Three layers yield one outer test cluster; the reported percentile bound
    # is then identically the point estimate and has no sampling uncertainty.
    minimal = common.synthetic_parity_streams(layers=3, experts=3, blocks_per_stream=2, seed=5, constrained=True)
    minimal_result = common.evaluate_nested(minimal, (common.Candidate("suffix", 2, 32),))
    uncertainty = minimal_result["paired_whole_layer_uncertainty"]
    if uncertainty["layers"] == 1 and uncertainty["lower_95_saving_bpw"] == uncertainty["point_saving_bpw"]:
        findings.append({
            "id": "DEGENERATE_ONE_CLUSTER_CONFIDENCE_GATE",
            "severity": "BLOCK",
            "detail": "The contract accepts >=3 layers, producing one test layer whose 4096 bootstrap replicas are identical; this is labeled a lower-95 bound and can gate promotion.",
        })

    # The control evaluator accepts prebuilt panels without a seed/provenance
    # field or a binding to source block moments.  This is a static interface
    # finding; no control payload is opened or generated here.
    import inspect
    control_signature = str(inspect.signature(common.evaluate_independent_matched_controls))
    control_source = inspect.getsource(common.evaluate_independent_matched_controls)
    if "control_panels" in control_signature and "matched_gaussian_raw_control" not in control_source:
        findings.append({
            "id": "GAUSSIAN_CONTROL_PROVENANCE_UNBOUND",
            "severity": "BLOCK",
            "detail": "evaluate_independent_matched_controls accepts arbitrary SymbolStream panels and assigns frozen seeds without verifying seed order, source geometry, block moments, or full raw-Gaussian relabel provenance.",
        })

    # A non-rectangular panel is accepted, so expert_group is not enforced as
    # one reusable slot universe across layers.
    irregular = tuple(tiny_stream(common, f"layer-{index // 3}", f"unique-{index}", index, symbols=64) for index in range(9))
    try:
        common.nested_partition(irregular)
    except Exception:
        irregular_accepted = False
    else:
        irregular_accepted = True
    if irregular_accepted:
        findings.append({
            "id": "EXPERT_SLOT_UNIVERSE_NOT_VALIDATED",
            "severity": "BLOCK",
            "detail": "nested_partition accepts an irregular panel with layer-unique expert IDs; it does not enforce the documented reusable whole-expert-slot grid.",
        })

    result = {
        "schema": "label-copula-census-independent-hostile-audit-v0",
        "status": "BLOCK_INDEPENDENT_SOURCE_REVIEW" if findings else "PASS_INDEPENDENT_SOURCE_REVIEW",
        "source_manifest_sha256": manifest_sha,
        "payloads_opened": 0,
        "current_codec_payloads_opened": 0,
        "gaussian_control_payloads_opened_or_generated": 0,
        "network_resources_opened": 0,
        "cupy_imported": False,
        "cuda_jobs": 0,
        "passes": passes,
        "findings": findings,
        "scope": {
            "diagnostic_labels_only": True,
            "transformed_strata_stream_deferred": True,
            "runtime": "Each frozen state recurrence is O(1), making each fixed-cell pass O(N); the 240-cell reference bank is O(240N).",
            "cupy_future_path": "No scientific CuPy implementation exists in this sealed source-only package; stage0 performs only a late CuPy availability import.",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
