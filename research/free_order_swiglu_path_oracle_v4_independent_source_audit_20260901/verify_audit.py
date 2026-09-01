#!/usr/bin/env python3
"""Pure-stdlib verifier for the independent FOSP-v4 BLOCK audit.

The verifier reads only the flat v4 source package, the two exact frozen v3
science files, and its own sealed artifacts.  It never follows model bindings
or imports a numerical/GPU/model package.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT.parent / "free_order_swiglu_path_oracle_v4"
V3 = ROOT.parent / "free_order_swiglu_path_oracle_v3"
AUDIT_MANIFEST = ROOT / "AUDIT_MANIFEST.json"

EXPECTED_PRODUCER: dict[str, tuple[int, str]] = {
    "PACKAGE_MANIFEST.json": (1674, "9762c7edffc86d21f4400d4fac37ecab33c75ede6f7f9ab7c3ef5b95fe51e066"),
    "PRODUCER_RECEIPT.json": (3149, "1366fc655527cd150ee6df22a8ad74c21a0b99bdee56c9bfc02c72cefbfb6fc4"),
    "README.md": (7150, "53dc4fb16900d28c96a47b39c0922c8a1566933ab99c7e4002addf14be991242"),
    "SOURCE_TEST_RECEIPT.json": (2384, "df57328f27bd2adf66b65401bf9f3a6a7f572a233cd177feb1d027946f83eb25"),
    "bootstrap_v4.py": (902, "46e91513325647de168cf0e0e39c5ab60988bd45cfcd2f8a4d6bb1a5c2d33544"),
    "launch_contract.py": (18059, "e331197f0e09768d2c85b2778c559bdd87a288c582981915d2dae2b2ca95aec6"),
    "native_launcher_contract.json": (5083, "fcd4cb53e7f01927f7ae3e9d71c99e27e90783786a43b5e15d76e4a4e1ed0811"),
    "scientific_oracle_v3.py": (21887, "9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070"),
    "scientific_protocol_v3.json": (7469, "f4660cb8876a749eb1635dbf010a8df6199e845b0517dd8b15039ac9cf1fd097"),
    "test_source_only.py": (12852, "7ec0f7499e62bbcc3795aa424ada3de1e0bc1aae9d729dea66b15e079518330f"),
    "verify_package.py": (8799, "5a2c0670b914841fa384937bfa68db7bcf1262e558d3ab06269cb70cef513c14"),
}
EXPECTED_V3 = {
    "free_order_oracle_v3.py": (21887, "9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070"),
    "protocol_lock.json": (7469, "f4660cb8876a749eb1635dbf010a8df6199e845b0517dd8b15039ac9cf1fd097"),
}
EXPECTED_STAGE_ORDER = [
    "direct_qwen_full3x3_pair_panel_including_legal_fp16",
    "gross_qwen_relaxed_reuse_necessary_bound",
    "eight_identically_processed_matched_controls_including_legal_fp16",
    "corrected_legal_fp16_statistic",
    "diagnostic_only_corrected_relaxed_statistic",
    "legal_fp16_survivor_or_optimization_gap",
]
EXPECTED_PRESTART_ORDER = [
    "external_launcher_regular_identity_and_sha256_verified",
    "launcher_authenticates_exact_contract_bytes",
    "launcher_authenticates_and_holds_bootstrap_bytes",
    "launcher_authenticates_complete_immutable_interpreter_runtime_image",
    "launcher_applies_environment_descriptor_and_namespace_closure",
    "launcher_creates_python_process_from_held_interpreter_identity",
]
HEAVY = {"numpy", "cupy", "torch", "scipy", "transformers", "safetensors", "cuda"}


class AuditFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        if not condition:
            raise AuditFailure(label)
        self.count += 1

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        self.require(observed == expected, f"{label}: {observed!r} != {expected!r}")

    def close(self, observed: float, expected: float, label: str, tol: float = 2e-15) -> None:
        self.require(math.isfinite(observed) and abs(observed - expected) <= tol,
                     f"{label}: {observed!r} != {expected!r}")

    def raises(self, exc_type: type[BaseException], call: Callable[[], Any], label: str) -> None:
        try:
            call()
        except exc_type:
            self.count += 1
            return
        raise AuditFailure(label + ": expected rejection")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path: Path, checks: Checks, label: str) -> bytes:
    info = path.lstat()
    checks.require(stat.S_ISREG(info.st_mode), label + " is regular")
    checks.require(not path.is_symlink(), label + " is not symlink")
    checks.equal(info.st_nlink, 1, label + " single link")
    raw = path.read_bytes()
    checks.equal(len(raw), info.st_size, label + " stable size")
    return raw


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditFailure("duplicate JSON key: " + key)
        result[key] = value
    return result


def _finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditFailure(label + " contains non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, label + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{label}[{index}]")


def parse_json(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuditFailure(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"invalid {label}: {exc}") from exc
    _finite(value, label)
    return value


def load_json(path: Path, checks: Checks, label: str) -> Any:
    return parse_json(regular_bytes(path, checks, label), label)


def direct_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def held_module(raw: bytes, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__["__file__"] = f"<held:{name}>"
    code = compile(raw, f"<held:{name}>", "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def verify_audit_closure(checks: Checks) -> tuple[dict[str, Any], str]:
    manifest_raw = regular_bytes(AUDIT_MANIFEST, checks, "audit manifest")
    manifest = parse_json(manifest_raw, "audit manifest")
    checks.equal(set(manifest), {"schema", "status", "artifact_count", "artifacts", "authorization"},
                 "audit manifest field closure")
    checks.equal(manifest["schema"], "free_order_swiglu_path_v4_independent_audit_manifest_v1",
                 "audit manifest schema")
    checks.equal(manifest["status"], "SEALED_BLOCK_AUDIT", "audit manifest status")
    checks.equal(manifest["authorization"], "NONE", "audit manifest authorization")
    rows = manifest["artifacts"]
    checks.equal(manifest["artifact_count"], len(rows), "audit artifact count")
    names = [row["path"] for row in rows]
    checks.equal(names, sorted(names), "audit manifest sorted paths")
    checks.equal(len(names), len(set(names)), "audit manifest unique paths")
    checks.equal(set(names), {"README.md", "replay_receipt.json", "verdict.json", "verify_audit.py"},
                 "audit manifest exact expected members")
    observed = sorted(entry.name for entry in os.scandir(ROOT))
    checks.equal(observed, sorted(names + ["AUDIT_MANIFEST.json"]), "audit exact object closure")
    for row in rows:
        checks.equal(set(row), {"path", "bytes", "sha256"}, "audit row closure")
        path = ROOT / row["path"]
        raw = regular_bytes(path, checks, "audit artifact " + row["path"])
        checks.equal(len(raw), row["bytes"], "audit artifact declared size " + row["path"])
        checks.equal(sha(raw), row["sha256"], "audit artifact declared hash " + row["path"])
    return manifest, sha(manifest_raw)


def verify_producer_closure(checks: Checks) -> dict[str, bytes]:
    checks.equal(sorted(entry.name for entry in os.scandir(PRODUCER)), sorted(EXPECTED_PRODUCER),
                 "producer exact object closure")
    output: dict[str, bytes] = {}
    for name, (size, digest) in EXPECTED_PRODUCER.items():
        raw = regular_bytes(PRODUCER / name, checks, "producer " + name)
        checks.equal(len(raw), size, "producer size " + name)
        checks.equal(sha(raw), digest, "producer hash " + name)
        output[name] = raw
    checks.equal(sum(len(raw) for raw in output.values()), 89408, "producer total bytes")

    manifest = parse_json(output["PACKAGE_MANIFEST.json"], "producer manifest")
    checks.equal(manifest["schema"], "free_order_swiglu_path_v4_package_manifest_v1",
                 "producer manifest schema")
    checks.equal(manifest["authorization"], "NONE", "producer manifest authority")
    rows = manifest["artifacts"]
    checks.equal(manifest["artifact_count"], 10, "producer manifest count")
    checks.equal([row["path"] for row in rows], sorted(set(EXPECTED_PRODUCER) - {"PACKAGE_MANIFEST.json"}),
                 "producer manifest exact path closure")
    for row in rows:
        checks.equal(set(row), {"path", "bytes", "sha256"}, "producer manifest row closure")
        checks.equal((row["bytes"], row["sha256"]), EXPECTED_PRODUCER[row["path"]],
                     "producer manifest row identity " + row["path"])
    for name, raw in output.items():
        if name.endswith(".py"):
            source = raw.decode("utf-8", errors="strict")
            ast.parse(source, filename=name)
            imports = direct_imports(source)
            checks.require(not (imports & HEAVY), "producer source has no heavy top import " + name)
            checks.require(not (imports & {"socket", "subprocess", "requests", "urllib", "ctypes"}),
                           "producer source has no network/process top import " + name)
    producer_receipt = parse_json(output["PRODUCER_RECEIPT.json"], "producer receipt")
    source_receipt = parse_json(output["SOURCE_TEST_RECEIPT.json"], "source test receipt")
    checks.require(all(value is False for value in producer_receipt["authorization"].values()),
                   "producer receipt grants no authority")
    checks.require(all(value is True for value in producer_receipt["deliberate_absences"].values()),
                   "producer receipt records all deployment absences")
    checks.require(all(value == 0 for value in source_receipt["zero_access_ledger"].values()),
                   "producer source-test zero-access ledger")
    checks.require(source_receipt["authorization"].startswith("NONE"),
                   "producer source-test grants no authority")
    return output


def verify_v3_identity(checks: Checks, producer: dict[str, bytes]) -> None:
    for name, (size, digest) in EXPECTED_V3.items():
        raw = regular_bytes(V3 / name, checks, "v3 producer " + name)
        checks.equal((len(raw), sha(raw)), (size, digest), "v3 exact identity " + name)
    checks.equal(producer["scientific_oracle_v3.py"], (V3 / "free_order_oracle_v3.py").read_bytes(),
                 "v4 oracle byte-identical to v3 producer")
    checks.equal(producer["scientific_protocol_v3.json"], (V3 / "protocol_lock.json").read_bytes(),
                 "v4 protocol byte-identical to v3 producer")


def verify_science(checks: Checks, producer: dict[str, bytes], verdict: dict[str, Any]) -> None:
    oracle_source = producer["scientific_oracle_v3.py"].decode("utf-8")
    protocol = parse_json(producer["scientific_protocol_v3.json"], "scientific protocol")
    oracle = held_module(producer["scientific_oracle_v3.py"], "fosp4_held_science")
    checks.equal(direct_imports(oracle_source), {"__future__", "math"}, "science top import closure")
    checks.equal(protocol["scientific_gate"]["stage_order"], EXPECTED_STAGE_ORDER,
                 "scientific stage order")
    checks.require(protocol["scientific_gate"]["gross_qwen_relaxed_reuse"]["only_permitted_family_hard_kill"] is True,
                   "gross relaxed only hard-kill declaration")
    checks.require(protocol["scientific_gate"]["corrected_relaxed_reuse"]["decision_eligible"] is False,
                   "corrected relaxed diagnostic only")
    tree = ast.parse(oracle_source)
    hard_kills = [node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and node.value.startswith("HARD_KILL")]
    checks.equal(hard_kills, ["HARD_KILL_GROSS_QWEN_RELAXED_NECESSARY_BOUND"],
                 "sole source hard-kill literal")
    legal_position = oracle_source.index('"legal_path_fp16": _controlled_statistic')
    relaxed_position = oracle_source.index('"relaxed_reuse_exact": _controlled_statistic')
    decision_position = oracle_source.index("decision = _decision_after_legal_statistics")
    checks.require(legal_position < relaxed_position < decision_position,
                   "legal FP16 statistic computed before diagnostic relaxed and decision")
    checks.require('statistics["relaxed_reuse_exact"]["decision_eligible"] = False' in oracle_source,
                   "corrected relaxed explicitly ineligible")
    checks.require("Gross Qwen target-wise reuse contains every legal exact path." in oracle_source,
                   "gross relaxed containment claim scoped")
    checks.require("optimization-gap ambiguity" in protocol["scientific_gate"]["achievable_path"]["failure_claim"],
                   "legal miss is optimization-gap ambiguity")

    with_missing: dict[str, Any] = {}
    qwen = {"legal_path_fp16": {"s_bpw": oracle.REQUIRED_GROSS_S + 0.1}}
    checks.raises(oracle.ProtocolError,
                  lambda: oracle._decision_after_legal_statistics(qwen, with_missing),
                  "decision refuses missing legal FP16")
    stats = {
        "legal_path_fp16": {"upper_confidence_survives_target": True},
        "relaxed_reuse_exact": {"upper_confidence_survives_target": False},
    }
    checks.equal(oracle._decision_after_legal_statistics(qwen, stats),
                 "SURVIVE_SOURCE_ORACLE_FP16_PATH_RESIDUAL_CODEC_REQUIRED",
                 "legal FP16 survivor decision")
    stats["relaxed_reuse_exact"]["upper_confidence_survives_target"] = True
    checks.equal(oracle._decision_after_legal_statistics(qwen, stats),
                 "SURVIVE_SOURCE_ORACLE_FP16_PATH_RESIDUAL_CODEC_REQUIRED",
                 "diagnostic relaxed cannot change decision")

    n = 8
    r = Fraction(7, 8)
    rho = r * r
    energy = Fraction(3 * n)
    metric = lambda capture: -0.5 * math.log2(float((energy - capture) / energy))
    q_relaxed = metric(Fraction(3 * n) * rho)
    c_relaxed = metric(Fraction(3 * n) * rho)
    q_legal = metric(Fraction(3 * (n - 1)) * rho)
    c_legal = metric(Fraction(3) * (2 * rho + (n - 3) * rho * rho))
    checks.equal(struct.unpack("<e", struct.pack("<e", float(r)))[0], float(r), "n8 r exact f16")
    checks.equal(struct.unpack("<e", struct.pack("<e", float(rho)))[0], float(rho), "n8 rho exact f16")
    independent = {
        "corrected_relaxed_s_bpw": q_relaxed - c_relaxed,
        "qwen_legal_fp16_s_bpw": q_legal,
        "control_legal_fp16_s_bpw": c_legal,
        "corrected_legal_fp16_s_bpw": q_legal - c_legal,
        "required_gross_s_bpw": oracle.REQUIRED_GROSS_S,
    }
    module_n8 = oracle.adversarial_n8_statistics()
    for key, expected in independent.items():
        checks.close(float(module_n8[key]), float(expected), "module n8 " + key)
        checks.close(float(verdict["science"]["n8_regression"][key]), float(expected),
                     "verdict n8 " + key)
    checks.require(q_legal - c_legal > oracle.REQUIRED_GROSS_S, "n8 legal survivor")


def verify_rate_read(checks: Checks, producer: dict[str, bytes], verdict: dict[str, Any]) -> None:
    oracle = held_module(producer["scientific_oracle_v3.py"], "fosp4_held_rate")
    weights = 3 * 768 * 2048
    info_bits = (math.factorial(768) - 1).bit_length()
    coefficient_count = 767 * 9
    side_bits = 64 * 8 + math.ceil(info_bits / 8) * 8 + coefficient_count * 16
    required_net = -0.5 * math.log2(0.8)
    side_bpw = side_bits / weights
    required_gross = required_net + side_bpw
    checks.equal(weights, 4718592, "weights per expert")
    checks.equal(info_bits, 6260, "factoradic information bits")
    checks.equal(math.ceil(info_bits / 8) * 8, 6264, "factoradic physical bits")
    checks.equal(coefficient_count, 6903, "coefficient count")
    checks.equal(side_bits, 117224, "total side bits")
    checks.close(side_bpw, 0.024843004014756944, "side bpw")
    checks.close(required_net, 0.16096404744368115, "required net")
    checks.close(required_gross, 0.1858070514584381, "required gross")
    maximum = 0.0
    for expected_row in verdict["rate_read"]["rows"]:
        rate = float(expected_row["requested_bpw"])
        frame = math.floor(weights * rate / 8)
        actual = frame * 8 / weights
        payload = actual - side_bpw
        cold = (math.ceil(frame / 4096) + 1) * 4096
        amplification = cold / frame
        maximum = max(maximum, amplification)
        checks.equal(frame, expected_row["frame_bytes"], f"frame bytes {rate}")
        checks.close(actual, expected_row["actual_bpw"], f"actual bpw {rate}")
        checks.close(payload, expected_row["residual_payload_bpw"], f"payload bpw {rate}")
        checks.equal(cold, expected_row["cold_page_bytes"], f"cold bytes {rate}")
        checks.close(amplification, expected_row["cold_page_amplification"], f"cold amp {rate}")
        frame_row = oracle.frame_ledger(rate)
        checks.equal(frame_row["frame_bytes"], frame, f"module frame bytes {rate}")
        checks.close(frame_row["cold_page_amplification"], amplification, f"module cold amp {rate}")
        checks.require(frame_row["strictly_below_2x"] is True, f"module read below 2x {rate}")
    checks.close(maximum, 1.0054349308378698, "maximum cold read")
    checks.close(verdict["rate_read"]["maximum_cold_page_amplification"], maximum,
                 "verdict maximum cold read")
    checks.equal(verdict["rate_read"]["logical_read_amplification"], 1.0,
                 "logical one-frame read")


def build_request(launch: types.ModuleType, contract_raw: bytes, bootstrap_raw: bytes) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    launcher_raw = b"synthetic native launcher fixture; never executed\n"
    runtime = {
        "bin/python3.12": b"synthetic held interpreter fixture; never executed\n",
        "lib/python3.12/codecs.py": b"# synthetic codecs\n",
        "lib/python3.12/encodings/__init__.py": b"# synthetic encodings\n",
        "lib/python3.12/io.py": b"# synthetic io\n",
    }
    members = [
        {
            "relative_path": path,
            "bytes": len(raw),
            "sha256": launch.sha256_bytes(raw),
            "object_type": "regular_file",
            "link_count": 1,
            "symlink_or_reparse": False,
            "owner_principal": "study_administrator",
            "immutable": True,
            "runtime_writable": False,
        }
        for path, raw in sorted(runtime.items())
    ]
    request = {
        "schema": launch.LAUNCH_REQUEST_SCHEMA,
        "status": "PRE_PYTHON_VALIDATION_ONLY_NO_EXECUTION",
        "contract_sha256": sha(contract_raw),
        "launcher": {
            "canonical_absolute_path": "/srv/fosp-v4-sealed/bin/fosp4-native-launcher",
            "bytes": len(launcher_raw),
            "sha256": sha(launcher_raw),
            "object_type": "regular_file",
            "link_count": 1,
            "symlink_or_reparse": False,
            "owner_principal": "study_administrator",
            "immutable": True,
            "external_pin_verified": True,
        },
        "bootstrap": {
            "relative_path": "bootstrap_v4.py",
            "bytes": len(bootstrap_raw),
            "sha256": sha(bootstrap_raw),
            "object_type": "regular_file",
            "link_count": 1,
            "symlink_or_reparse": False,
            "authenticated_and_held_before_execution": True,
        },
        "runtime_image": {
            "schema": launch.RUNTIME_IMAGE_SCHEMA,
            "image_id": launch.runtime_image_id(members),
            "image_kind": "complete_immutable_authenticated_interpreter_runtime",
            "administrator_principal": "study_administrator",
            "runtime_principal": "study_runtime",
            "ordinary_mutable_venv": False,
            "authenticated_before_python_startup": True,
            "platform_immutable": True,
            "runtime_write_unlink_relabel_alias_capability": False,
            "complete_object_closure": True,
            "members": members,
        },
        "pre_start_order": list(EXPECTED_PRESTART_ORDER),
        "python_process_created": False,
        "authorization": False,
    }
    return request, launcher_raw, runtime


def run_request(launch: types.ModuleType, contract: dict[str, Any], contract_raw: bytes,
                bootstrap_raw: bytes, request: dict[str, Any], launcher_raw: bytes,
                runtime: dict[str, bytes], *, bootstrap_override: bytes | None = None,
                launcher_pin: str | None = None) -> dict[str, Any]:
    return launch.validate_pre_python_launch_request(
        contract,
        contract_raw,
        launch.canonical_json_bytes(request),
        external_contract_sha256=sha(contract_raw),
        external_launcher_sha256=launcher_pin or sha(launcher_raw),
        launcher_raw=launcher_raw,
        bootstrap_raw=bootstrap_raw if bootstrap_override is None else bootstrap_override,
        runtime_member_bytes=runtime,
    )


def verify_contract_and_exploits(checks: Checks, producer: dict[str, bytes],
                                 verdict: dict[str, Any], replay: dict[str, Any]) -> None:
    contract_raw = producer["native_launcher_contract.json"]
    bootstrap_raw = producer["bootstrap_v4.py"]
    contract_json = parse_json(contract_raw, "native launcher contract")
    launch_source = producer["launch_contract.py"].decode("utf-8")
    launch = held_module(producer["launch_contract.py"], "fosp4_held_contract")
    contract = dict(launch.validate_contract(
        contract_raw,
        bootstrap_raw,
        producer["scientific_oracle_v3.py"],
        producer["scientific_protocol_v3.json"],
    ))
    checks.equal(contract_json["mandatory_pre_start_order"], EXPECTED_PRESTART_ORDER,
                 "static prestart order")
    checks.equal(contract["mandatory_pre_start_order"], EXPECTED_PRESTART_ORDER,
                 "validated prestart order")
    checks.equal(contract["bootstrap_subject"]["sha256"], sha(bootstrap_raw),
                 "bootstrap hash pinned by contract")
    checks.require(contract["bootstrap_subject"]["exact_bytes_authenticated_and_held_before_execution"] is True,
                   "bootstrap held declaration")
    checks.require(contract["external_native_launcher"]["included_in_source_package"] is False,
                   "native launcher absent")
    checks.require(contract["source_default"]["interpreter_image_present"] is False,
                   "runtime image absent")
    checks.require(contract["source_default"]["runtime_manifest_present"] is False,
                   "runtime manifest absent")
    checks.require(all(value is False for value in contract["authorization"].values()),
                   "contract grants no authority")
    checks.require(all(value == 0 for value in contract["zero_access"].values()),
                   "contract zero-access ledger")
    checks.require(direct_imports(launch_source) <= {"__future__", "hashlib", "json", "re", "pathlib", "typing"},
                   "launch model pure stdlib imports")
    checks.require(not (direct_imports(launch_source) & HEAVY), "launch model no heavy imports")

    bootstrap = held_module(bootstrap_raw, "fosp4_held_bootstrap")
    checks.require(bootstrap.AUTHORITY_GRANTED is False, "bootstrap authority false")
    checks.require(78 in bootstrap._main.__code__.co_consts, "bootstrap terminal refusal code")
    checks.require(bootstrap.source_only_status()["authorization"] is False,
                   "bootstrap source status no authority")

    readme = producer["README.md"].decode("utf-8")
    checks.require("/srv/fosp-v4-sealed/bin/fosp4-native-launcher" in readme,
                   "README absolute illustrative native launcher")
    checks.require(re.search(r"(^|\s)python\s+-", readme, flags=re.MULTILINE) is None,
                   "README no generic python invocation")
    checks.require("ordinary mutable virtual environment does **not** satisfy" in readme,
                   "README rejects mutable venv")
    checks.require("This package explicitly authorizes nothing" in readme,
                   "README no authority")

    request, launcher_raw, runtime = build_request(launch, contract_raw, bootstrap_raw)
    baseline = run_request(launch, contract, contract_raw, bootstrap_raw,
                           request, launcher_raw, runtime)
    checks.equal(baseline["status"], "PASS_CONTRACT_VALIDATION_ONLY", "baseline synthetic request")
    checks.require(baseline["python_process_created"] is False and baseline["authorization"] is False,
                   "baseline is inert")

    checks.raises(launch.ContractViolation,
                  lambda: run_request(launch, contract, contract_raw, bootstrap_raw,
                                      request, launcher_raw, runtime,
                                      bootstrap_override=b"hostile bootstrap"),
                  "inconsistent bootstrap substitution rejected")
    changed_runtime = dict(runtime)
    changed_runtime["bin/python3.12"] = b"changed without manifest update"
    checks.raises(launch.ContractViolation,
                  lambda: run_request(launch, contract, contract_raw, bootstrap_raw,
                                      request, launcher_raw, changed_runtime),
                  "inconsistent runtime substitution rejected")
    checks.raises(launch.ContractViolation,
                  lambda: run_request(launch, contract, contract_raw, bootstrap_raw,
                                      request, launcher_raw, runtime, launcher_pin="00" * 32),
                  "external launcher pin mismatch rejected")
    request["launcher"]["symlink_or_reparse"] = True
    checks.raises(launch.ContractViolation,
                  lambda: run_request(launch, contract, contract_raw, bootstrap_raw,
                                      request, launcher_raw, runtime),
                  "declared launcher symlink rejected")

    # Blocking exploit 1: the hostile runtime mints the only digest used to
    # authenticate itself.  No external runtime identity enters the API.
    request, launcher_raw, runtime = build_request(launch, contract_raw, bootstrap_raw)
    hostile = b"hostile self-described interpreter; would execute before bootstrap\n"
    runtime["bin/python3.12"] = hostile
    row = next(item for item in request["runtime_image"]["members"]
               if item["relative_path"] == "bin/python3.12")
    row["bytes"] = len(hostile)
    row["sha256"] = sha(hostile)
    request["runtime_image"]["image_id"] = launch.runtime_image_id(request["runtime_image"]["members"])
    rebound = run_request(launch, contract, contract_raw, bootstrap_raw,
                          request, launcher_raw, runtime)
    checks.equal(rebound["status"], "PASS_CONTRACT_VALIDATION_ONLY",
                 "self-rebound hostile runtime accepted")
    argument_names = set(launch.validate_pre_python_launch_request.__code__.co_varnames[
        :launch.validate_pre_python_launch_request.__code__.co_argcount
         + launch.validate_pre_python_launch_request.__code__.co_kwonlyargcount
    ])
    checks.require("external_runtime_sha256" not in argument_names and
                   "external_runtime_manifest_sha256" not in argument_names and
                   "external_runtime_image_id" not in argument_names,
                   "validator has no external runtime trust argument")
    checks.require("external_runtime_manifest_sha256" not in contract_json["interpreter_runtime"] and
                   "external_runtime_image_id" not in contract_json["interpreter_runtime"],
                   "pinned contract has no external runtime identity")
    checks.equal(verdict["blocking_findings"]["FOSP4-FW-001"]["observed_status"],
                 rebound["status"], "runtime blocker receipt")

    # Blocking exploit 2: filename heuristics accept a shell and a path that
    # still contains a parent traversal component as "canonical".
    request, launcher_raw, runtime = build_request(launch, contract_raw, bootstrap_raw)
    request["launcher"]["canonical_absolute_path"] = "/bin/bash"
    shell_result = run_request(launch, contract, contract_raw, bootstrap_raw,
                               request, launcher_raw, runtime)
    checks.equal(shell_result["status"], "PASS_CONTRACT_VALIDATION_ONLY",
                 "bash launcher accepted")
    request, launcher_raw, runtime = build_request(launch, contract_raw, bootstrap_raw)
    request["launcher"]["canonical_absolute_path"] = "/srv/fosp-v4/../bin/fosp4-native-launcher"
    dotdot_result = run_request(launch, contract, contract_raw, bootstrap_raw,
                                request, launcher_raw, runtime)
    checks.equal(dotdot_result["status"], "PASS_CONTRACT_VALIDATION_ONLY",
                 "dotdot launcher accepted as canonical")
    checks.equal(verdict["blocking_findings"]["FOSP4-FW-003"]["shell_fixture_status"],
                 shell_result["status"], "shell blocker receipt")
    checks.equal(verdict["blocking_findings"]["FOSP4-FW-003"]["non_normal_fixture_status"],
                 dotdot_result["status"], "dotdot blocker receipt")

    checks.require("entrypoint_exec_source" not in launch.REQUEST_KEYS,
                   "request has no held bootstrap execution source binding")
    checks.require(verdict["pre_python_contract"]["bootstrap_actual_execution_from_held_identity_evidenced"] is False,
                   "held bootstrap execution is not evidenced")
    checks.require(verdict["pre_python_contract"]["native_launcher_implementation_audited"] is False,
                   "native implementation not audited")
    checks.equal(replay["hostile_fixture_matrix"]["runtime_bytes_changed_with_self_consistent_untrusted_manifest"],
                 "BLOCK_ACCEPTED", "replay runtime exploit")
    checks.equal(replay["hostile_fixture_matrix"]["bash_launcher"], "BLOCK_ACCEPTED",
                 "replay shell exploit")


def verify_receipts(checks: Checks, verdict: dict[str, Any], replay: dict[str, Any]) -> None:
    checks.equal(verdict["schema"], "free_order_swiglu_path_v4_independent_source_audit_verdict_v1",
                 "verdict schema")
    checks.equal(verdict["status"], "BLOCK_RUNTIME_TRUST_AND_NATIVE_ENFORCEMENT_INCOMPLETE",
                 "verdict status")
    checks.equal(verdict["verdict"], "BLOCK", "verdict BLOCK")
    checks.require(verdict["producer_modified"] is False, "producer unmodified receipt")
    checks.require(verdict["v3_audit_evidence_used"] is False, "no v3 audit evidence")
    checks.equal(set(verdict["blocking_findings"]), {"FOSP4-FW-001", "FOSP4-FW-002", "FOSP4-FW-003"},
                 "blocking finding closure")
    checks.equal(verdict["blocking_decision"], {
        "source_or_model_access": "BLOCKED",
        "runtime_calibration": "BLOCKED",
        "gpu_experiment": "BLOCKED",
        "deployment": "BLOCKED",
        "production_authorization": "BLOCKED",
        "successor_requirement": "A distinct successor must externally pin the runtime identity, supply and independently audit the native launcher/runtime, bind held bootstrap execution, and reject shell/non-normal launcher identities before any payload access",
    }, "blocking decision")
    checks.equal(replay["schema"], "free_order_swiglu_path_v4_independent_source_replay_receipt_v1",
                 "replay schema")
    checks.equal(replay["status"], "SOURCE_ONLY_REPLAY_COMPLETE_BLOCK_CONFIRMED", "replay status")
    checks.equal(replay["linux_runpod"]["producer_verifier_checks"], 105, "RunPod producer checks")
    checks.equal(replay["linux_runpod"]["producer_tests_run"], 18, "RunPod tests run")
    checks.equal(replay["linux_runpod"]["producer_tests_passed"], 18, "RunPod tests pass")
    checks.require(replay["linux_runpod"]["manifest_matches_local"] is True, "RunPod manifest matches")
    checks.equal(replay["authorization"], "NONE", "replay authority none")
    for key, value in verdict["zero_access_ledger"].items():
        checks.equal(value, 0, "zero access " + key)
    checks.require(not (set(sys.modules) & HEAVY), "audit imported no heavy modules")


def verify() -> dict[str, Any]:
    checks = Checks()
    manifest, manifest_sha = verify_audit_closure(checks)
    producer = verify_producer_closure(checks)
    verify_v3_identity(checks, producer)
    verdict = load_json(ROOT / "verdict.json", checks, "verdict")
    replay = load_json(ROOT / "replay_receipt.json", checks, "replay receipt")
    verify_receipts(checks, verdict, replay)
    verify_science(checks, producer, verdict)
    verify_rate_read(checks, producer, verdict)
    verify_contract_and_exploits(checks, producer, verdict, replay)
    checks.equal(manifest["authorization"], "NONE", "final manifest authority none")
    return {
        "schema": "free_order_swiglu_path_v4_independent_audit_verification_v1",
        "status": "BLOCK_CONFIRMED",
        "verdict": "BLOCK",
        "checks": checks.count,
        "audit_manifest_sha256": manifest_sha,
        "producer_manifest_sha256": EXPECTED_PRODUCER["PACKAGE_MANIFEST.json"][1],
        "science": "PASS_FROZEN_V3",
        "rate_read": "PASS_1X_LOGICAL_1.0054349308378698X_MAX_COLD",
        "blocking_findings": ["FOSP4-FW-001", "FOSP4-FW-002", "FOSP4-FW-003"],
        "model_or_qwen_access": 0,
        "gpu_operations": 0,
        "network_operations": 0,
        "authorizations_issued": 0,
        "authorization": "NONE",
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise
