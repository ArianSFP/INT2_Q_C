"""Inert validation model for the FOSP-v4 pre-Python native-launch boundary."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


AUTHORITY_GRANTED = False
LAUNCH_REQUEST_SCHEMA = "free_order_swiglu_path_pre_python_launch_request_v4"
RUNTIME_IMAGE_SCHEMA = "free_order_swiglu_path_immutable_interpreter_image_v4"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

CONTRACT_KEYS = frozenset(
    {
        "schema",
        "status",
        "lineage",
        "frozen_science",
        "external_native_launcher",
        "bootstrap_subject",
        "interpreter_runtime",
        "mandatory_pre_start_order",
        "source_default",
        "zero_access",
        "authorization",
    }
)
REQUEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "contract_sha256",
        "launcher",
        "bootstrap",
        "runtime_image",
        "pre_start_order",
        "python_process_created",
        "authorization",
    }
)
LAUNCHER_KEYS = frozenset(
    {
        "canonical_absolute_path",
        "bytes",
        "sha256",
        "object_type",
        "link_count",
        "symlink_or_reparse",
        "owner_principal",
        "immutable",
        "external_pin_verified",
    }
)
BOOTSTRAP_KEYS = frozenset(
    {
        "relative_path",
        "bytes",
        "sha256",
        "object_type",
        "link_count",
        "symlink_or_reparse",
        "authenticated_and_held_before_execution",
    }
)
RUNTIME_KEYS = frozenset(
    {
        "schema",
        "image_id",
        "image_kind",
        "administrator_principal",
        "runtime_principal",
        "ordinary_mutable_venv",
        "authenticated_before_python_startup",
        "platform_immutable",
        "runtime_write_unlink_relabel_alias_capability",
        "complete_object_closure",
        "members",
    }
)
MEMBER_KEYS = frozenset(
    {
        "relative_path",
        "bytes",
        "sha256",
        "object_type",
        "link_count",
        "symlink_or_reparse",
        "owner_principal",
        "immutable",
        "runtime_writable",
    }
)

EXPECTED_STAGE_ORDER = [
    "direct_qwen_full3x3_pair_panel_including_legal_fp16",
    "gross_qwen_relaxed_reuse_necessary_bound",
    "eight_identically_processed_matched_controls_including_legal_fp16",
    "corrected_legal_fp16_statistic",
    "diagnostic_only_corrected_relaxed_statistic",
    "legal_fp16_survivor_or_optimization_gap",
]
EXPECTED_PRE_START_ORDER = [
    "external_launcher_regular_identity_and_sha256_verified",
    "launcher_authenticates_exact_contract_bytes",
    "launcher_authenticates_and_holds_bootstrap_bytes",
    "launcher_authenticates_complete_immutable_interpreter_runtime_image",
    "launcher_applies_environment_descriptor_and_namespace_closure",
    "launcher_creates_python_process_from_held_interpreter_identity",
]
REQUIRED_STARTUP_MEMBERS = {
    "bin/python3.12",
    "lib/python3.12/codecs.py",
    "lib/python3.12/encodings/__init__.py",
    "lib/python3.12/io.py",
}


class ContractViolation(ValueError):
    pass


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f"not canonical JSON: {exc}") from exc


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str, *, canonical: bool = False) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractViolation(f"non-finite number in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation(f"invalid {label}: {exc}") from exc
    if canonical and canonical_json_bytes(value) != raw:
        raise ContractViolation(f"{label} is not canonical JSON")
    return value


def _exact(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractViolation(f"{label} must be an object")
    observed = frozenset(value)
    if observed != keys:
        raise ContractViolation(
            f"{label} field closure mismatch; missing={sorted(keys-observed)}, extra={sorted(observed-keys)}"
        )
    return value


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise ContractViolation(label)


def _false(value: Any, label: str) -> None:
    if value is not False:
        raise ContractViolation(f"{label} must be false")


def _true(value: Any, label: str) -> None:
    if value is not True:
        raise ContractViolation(f"{label} must be true")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ContractViolation(f"{label} must be lowercase SHA-256")
    return value


def _size(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolation(f"{label} must be a nonnegative integer")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ContractViolation(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {"", "."} or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractViolation(f"{label} must be normalized and contained")
    return value


def _verify_bytes(raw: bytes, size: Any, digest: Any, label: str) -> None:
    expected_size = _size(size, label + ".bytes")
    expected_digest = _sha(digest, label + ".sha256")
    _require(len(raw) == expected_size and sha256_bytes(raw) == expected_digest, label + " byte identity mismatch")


def validate_contract(
    contract_raw: bytes,
    bootstrap_raw: bytes,
    scientific_oracle_raw: bytes,
    scientific_protocol_raw: bytes,
) -> Mapping[str, Any]:
    contract = _exact(parse_json(contract_raw, "native launcher contract"), CONTRACT_KEYS, "contract")
    _require(contract["schema"] == "free_order_swiglu_path_native_launcher_contract_v4", "contract schema")
    _require(contract["status"] == "SOURCE_ONLY_CONTRACT_NATIVE_LAUNCHER_NOT_INCLUDED", "contract status")

    science = contract["frozen_science"]
    _require(science["stage_order"] == EXPECTED_STAGE_ORDER, "v3 scientific order changed")
    _require(science["scientific_semantics_changed_from_v3"] is False, "science change flag")
    _require(science["roles"] == ["gate", "up", "down_transposed"], "role order changed")
    _require(science["coefficients_per_edge"] == 9 and science["path_edges"] == 767, "path science changed")
    _require(science["total_side_bits"] == 117224, "side bits changed")
    _require(science["total_side_bpw"] == 0.024843004014756944, "side rate changed")
    _require(science["required_net_s_bpw"] == 0.16096404744368115, "net rate changed")
    _require(science["required_gross_s_bpw"] == 0.1858070514584381, "gross rate changed")
    _require(science["logical_read_amplification"] == 1.0, "logical read changed")
    _require(science["maximum_cold_page_amplification"] == 1.0054349308378698, "cold read changed")
    n8 = science["n8_regression"]
    _require(
        n8
        == {
            "n": 8,
            "r": 0.875,
            "rho": 0.765625,
            "corrected_relaxed_s_bpw": 0.0,
            "qwen_legal_fp16_s_bpw": 0.7995602818589078,
            "control_legal_fp16_s_bpw": 0.5885652320580218,
            "corrected_legal_fp16_s_bpw": 0.21099504980088601,
            "required_gross_s_bpw": 0.1858070514584381,
        },
        "n=8 regression changed",
    )
    _verify_bytes(
        scientific_oracle_raw,
        len(scientific_oracle_raw),
        science["scientific_oracle_v3_sha256"],
        "scientific oracle",
    )
    _verify_bytes(
        scientific_protocol_raw,
        len(scientific_protocol_raw),
        science["scientific_protocol_v3_sha256"],
        "scientific protocol",
    )

    launcher = contract["external_native_launcher"]
    for field in (
        "external_sha256_pin_required",
        "canonical_absolute_path_required",
        "regular_file_required",
        "single_link_required",
        "symlink_or_reparse_forbidden",
        "administrator_owned_immutable_required",
        "self_identity_verified_before_opening_contract",
        "ordinary_python_or_shell_launcher_forbidden",
    ):
        _true(launcher[field], "external_native_launcher." + field)
    _false(launcher["included_in_source_package"], "launcher included")
    _require(launcher["implementation_language_class"] == "native_no_python_runtime_dependency", "native launcher class")

    bootstrap = contract["bootstrap_subject"]
    _verify_bytes(bootstrap_raw, bootstrap["bytes"], bootstrap["sha256"], "bootstrap subject")
    _require(bootstrap["relative_path"] == "bootstrap_v4.py", "bootstrap path")
    _true(bootstrap["opened_no_follow_as_regular_single_link_file"], "bootstrap no-follow")
    _true(bootstrap["exact_bytes_authenticated_and_held_before_execution"], "bootstrap pre-exec auth")
    _true(bootstrap["python_self_authentication_is_not_accepted"], "no Python self-auth")

    runtime = contract["interpreter_runtime"]
    for field in (
        "complete_image_manifest_required",
        "complete_exact_byte_and_object_closure_required",
        "image_authenticated_before_python_process_creation",
        "interpreter_opened_no_follow_and_executed_from_held_identity",
        "all_startup_modules_and_encoding_resources_in_image",
        "startup_files_authenticated_before_any_python_startup_code",
        "single_link_regular_members_required",
        "administrator_owned_platform_immutability_required",
        "runtime_user_has_no_write_unlink_relabel_or_alias_capability",
        "administrator_compromise_out_of_scope",
    ):
        _true(runtime[field], "interpreter_runtime." + field)
    _false(runtime["ordinary_mutable_venv_satisfies_contract"], "ordinary mutable venv")
    _require(set(runtime["required_startup_members"]) == REQUIRED_STARTUP_MEMBERS, "startup member closure")
    _require(contract["mandatory_pre_start_order"] == EXPECTED_PRE_START_ORDER, "pre-start order changed")
    _false(contract["source_default"]["launcher_present"], "source launcher presence")
    _false(contract["source_default"]["interpreter_image_present"], "source interpreter presence")
    _false(contract["source_default"]["runtime_manifest_present"], "source runtime manifest presence")
    _false(contract["source_default"]["signature_or_authorization_builder_present"], "authorization builder")
    _require(all(value == 0 for value in contract["zero_access"].values()), "zero-access ledger not zero")
    _require(all(value is False for value in contract["authorization"].values()), "contract grants authority")
    return contract


def runtime_image_id(members: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {"bytes": row["bytes"], "relative_path": row["relative_path"], "sha256": row["sha256"]}
        for row in members
    ]
    return sha256_bytes(canonical_json_bytes({"members": identity, "schema": RUNTIME_IMAGE_SCHEMA}))


def validate_pre_python_launch_request(
    contract: Mapping[str, Any],
    contract_raw: bytes,
    request_raw: bytes,
    *,
    external_contract_sha256: str,
    external_launcher_sha256: str,
    launcher_raw: bytes,
    bootstrap_raw: bytes,
    runtime_member_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate a synthetic pre-start request; never start Python or grant authority."""

    _require(sha256_bytes(contract_raw) == _sha(external_contract_sha256, "external contract pin"), "contract substitution before native launcher")
    request = _exact(parse_json(request_raw, "launch request", canonical=True), REQUEST_KEYS, "launch request")
    _require(request["schema"] == LAUNCH_REQUEST_SCHEMA, "launch request schema")
    _require(request["status"] == "PRE_PYTHON_VALIDATION_ONLY_NO_EXECUTION", "launch request status")
    _require(request["contract_sha256"] == external_contract_sha256, "launch request contract binding")
    _false(request["python_process_created"], "Python process created")
    _false(request["authorization"], "launch request authorization")

    launcher = _exact(request["launcher"], LAUNCHER_KEYS, "launcher")
    path = launcher["canonical_absolute_path"]
    _require(isinstance(path, str) and path.startswith("/") and PurePosixPath(path).as_posix() == path, "launcher canonical absolute path")
    _require("python" not in PurePosixPath(path).name.lower() and "sh" != PurePosixPath(path).name.lower(), "generic Python or shell launcher forbidden")
    _verify_bytes(launcher_raw, launcher["bytes"], launcher["sha256"], "native launcher")
    _require(launcher["sha256"] == _sha(external_launcher_sha256, "external launcher pin"), "native launcher external pin mismatch")
    _require(launcher["object_type"] == "regular_file", "native launcher must be regular")
    _require(launcher["link_count"] == 1, "native launcher must be single-link")
    _false(launcher["symlink_or_reparse"], "native launcher symlink/reparse")
    _require(launcher["owner_principal"] == "study_administrator", "native launcher owner")
    _true(launcher["immutable"], "native launcher immutable")
    _true(launcher["external_pin_verified"], "native launcher external pin not verified")

    bootstrap = _exact(request["bootstrap"], BOOTSTRAP_KEYS, "bootstrap")
    _require(bootstrap["relative_path"] == contract["bootstrap_subject"]["relative_path"], "bootstrap path mismatch")
    _verify_bytes(bootstrap_raw, bootstrap["bytes"], bootstrap["sha256"], "bootstrap")
    _require(bootstrap["sha256"] == contract["bootstrap_subject"]["sha256"], "bootstrap substitution before execution")
    _require(bootstrap["object_type"] == "regular_file" and bootstrap["link_count"] == 1, "bootstrap not regular single-link")
    _false(bootstrap["symlink_or_reparse"], "bootstrap symlink/reparse")
    _true(bootstrap["authenticated_and_held_before_execution"], "bootstrap not authenticated before execution")

    image = _exact(request["runtime_image"], RUNTIME_KEYS, "runtime image")
    _require(image["schema"] == RUNTIME_IMAGE_SCHEMA, "runtime image schema")
    _require(image["image_kind"] == "complete_immutable_authenticated_interpreter_runtime", "ordinary mutable venv or incomplete runtime forbidden")
    _require(image["administrator_principal"] == "study_administrator", "runtime administrator")
    _require(image["runtime_principal"] == "study_runtime", "runtime principal")
    _require(image["administrator_principal"] != image["runtime_principal"], "administrator/runtime separation")
    _false(image["ordinary_mutable_venv"], "ordinary mutable venv forbidden")
    _true(image["authenticated_before_python_startup"], "runtime must be authenticated before Python startup")
    _true(image["platform_immutable"], "runtime image must be platform immutable")
    _false(image["runtime_write_unlink_relabel_alias_capability"], "runtime mutation capability")
    _true(image["complete_object_closure"], "runtime object closure incomplete")

    members = image["members"]
    _require(isinstance(members, list) and bool(members), "runtime members must be nonempty")
    paths: list[str] = []
    normalized: list[Mapping[str, Any]] = []
    for index, value in enumerate(members):
        row = _exact(value, MEMBER_KEYS, f"runtime member {index}")
        member_path = _relative(row["relative_path"], f"runtime member {index} path")
        paths.append(member_path)
        if member_path not in runtime_member_bytes:
            raise ContractViolation("missing authenticated pre-startup runtime bytes: " + member_path)
        _verify_bytes(runtime_member_bytes[member_path], row["bytes"], row["sha256"], f"runtime member {member_path}")
        _require(row["object_type"] == "regular_file", "runtime member not regular: " + member_path)
        _require(row["link_count"] == 1, "runtime member not single-link: " + member_path)
        _false(row["symlink_or_reparse"], "runtime member link/reparse: " + member_path)
        _require(row["owner_principal"] == image["administrator_principal"], "runtime member owner: " + member_path)
        _true(row["immutable"], "mutable runtime member: " + member_path)
        _false(row["runtime_writable"], "runtime-writable member: " + member_path)
        normalized.append(row)
    _require(paths == sorted(paths) and len(paths) == len(set(paths)), "runtime paths must be sorted and unique")
    _require(set(paths) == set(runtime_member_bytes), "runtime member exact-byte closure mismatch")
    _require(REQUIRED_STARTUP_MEMBERS <= set(paths), "missing authenticated pre-startup runtime member")
    _require(image["image_id"] == runtime_image_id(normalized), "runtime image identity mismatch")
    _require(request["pre_start_order"] == EXPECTED_PRE_START_ORDER, "pre-start order mismatch")
    return {
        "schema": "free_order_swiglu_path_v4_pre_start_validation_result",
        "status": "PASS_CONTRACT_VALIDATION_ONLY",
        "python_process_created": False,
        "model_or_qwen_access": 0,
        "gpu_operations": 0,
        "authorization": False,
    }


def _main() -> int:
    print("FOSP-v4 launch-contract validator is inert and authorizes nothing.")
    return 78


if __name__ == "__main__":
    raise SystemExit(_main())
