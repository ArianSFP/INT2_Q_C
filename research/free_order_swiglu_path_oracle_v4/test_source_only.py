"""Hostile standard-library tests for the inert FOSP-v4 launch contract."""

from __future__ import annotations

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bootstrap_v4
import launch_contract as launch


def package_inputs() -> tuple[bytes, bytes, bytes, bytes]:
    return (
        (ROOT / "native_launcher_contract.json").read_bytes(),
        (ROOT / "bootstrap_v4.py").read_bytes(),
        (ROOT / "scientific_oracle_v3.py").read_bytes(),
        (ROOT / "scientific_protocol_v3.json").read_bytes(),
    )


def validated_contract() -> tuple[dict[str, object], bytes, bytes]:
    contract_raw, bootstrap_raw, oracle_raw, protocol_raw = package_inputs()
    contract = launch.validate_contract(contract_raw, bootstrap_raw, oracle_raw, protocol_raw)
    return dict(contract), contract_raw, bootstrap_raw


def build_request() -> tuple[dict[str, object], bytes, bytes, dict[str, bytes]]:
    contract, contract_raw, bootstrap_raw = validated_contract()
    launcher_raw = b"synthetic native launcher identity fixture; never executed\n"
    runtime_bytes = {
        "bin/python3.12": b"synthetic held interpreter fixture; never executed\n",
        "lib/python3.12/codecs.py": b"# synthetic immutable codecs fixture\n",
        "lib/python3.12/encodings/__init__.py": b"# synthetic immutable encodings fixture\n",
        "lib/python3.12/io.py": b"# synthetic immutable io fixture\n",
    }
    members: list[dict[str, object]] = []
    for path, raw in sorted(runtime_bytes.items()):
        members.append(
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
        )
    request: dict[str, object] = {
        "schema": launch.LAUNCH_REQUEST_SCHEMA,
        "status": "PRE_PYTHON_VALIDATION_ONLY_NO_EXECUTION",
        "contract_sha256": launch.sha256_bytes(contract_raw),
        "launcher": {
            "canonical_absolute_path": "/srv/fosp-v4-sealed/bin/fosp4-native-launcher",
            "bytes": len(launcher_raw),
            "sha256": launch.sha256_bytes(launcher_raw),
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
            "sha256": launch.sha256_bytes(bootstrap_raw),
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
        "pre_start_order": list(launch.EXPECTED_PRE_START_ORDER),
        "python_process_created": False,
        "authorization": False,
    }
    return request, contract_raw, launcher_raw, runtime_bytes


def validate_request(
    request: dict[str, object],
    contract_raw: bytes,
    launcher_raw: bytes,
    runtime_bytes: dict[str, bytes],
    *,
    bootstrap_raw: bytes | None = None,
    external_launcher_sha256: str | None = None,
) -> dict[str, object]:
    contract, _, genuine_bootstrap = validated_contract()
    return launch.validate_pre_python_launch_request(
        contract,
        contract_raw,
        launch.canonical_json_bytes(request),
        external_contract_sha256=launch.sha256_bytes(contract_raw),
        external_launcher_sha256=external_launcher_sha256 or launch.sha256_bytes(launcher_raw),
        launcher_raw=launcher_raw,
        bootstrap_raw=genuine_bootstrap if bootstrap_raw is None else bootstrap_raw,
        runtime_member_bytes=runtime_bytes,
    )


class FrozenScienceTests(unittest.TestCase):
    def test_exact_v3_contract_and_science_bindings(self) -> None:
        contract, _, _ = validated_contract()
        science = contract["frozen_science"]
        self.assertFalse(science["scientific_semantics_changed_from_v3"])
        self.assertEqual(science["stage_order"], launch.EXPECTED_STAGE_ORDER)
        self.assertEqual(science["total_side_bits"], 117224)
        self.assertEqual(science["required_gross_s_bpw"], 0.1858070514584381)
        self.assertEqual(science["maximum_cold_page_amplification"], 1.0054349308378698)

    def test_exact_n8_regression(self) -> None:
        contract, _, _ = validated_contract()
        n8 = contract["frozen_science"]["n8_regression"]
        self.assertEqual(n8["corrected_relaxed_s_bpw"], 0.0)
        self.assertEqual(n8["corrected_legal_fp16_s_bpw"], 0.21099504980088601)
        self.assertGreater(n8["corrected_legal_fp16_s_bpw"], n8["required_gross_s_bpw"])

    def test_duplicate_contract_key_rejected(self) -> None:
        with self.assertRaisesRegex(launch.ContractViolation, "duplicate JSON key"):
            launch.parse_json(b'{"schema":1,"schema":2}', "duplicate contract")


class HostilePreStartupTests(unittest.TestCase):
    def test_request_serialization_round_trip(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        raw = launch.canonical_json_bytes(request)
        self.assertEqual(launch.parse_json(raw, "request", canonical=True), request)
        result = validate_request(request, contract_raw, launcher_raw, runtime_bytes)
        self.assertEqual(result["status"], "PASS_CONTRACT_VALIDATION_ONLY")
        self.assertFalse(result["python_process_created"])
        self.assertFalse(result["authorization"])

    def test_hostile_bootstrap_substitution_rejected_before_execution(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        with self.assertRaisesRegex(launch.ContractViolation, "bootstrap byte identity mismatch"):
            validate_request(
                request,
                contract_raw,
                launcher_raw,
                runtime_bytes,
                bootstrap_raw=b"print('hostile bootstrap executed')\n",
            )

    def test_hostile_native_launcher_substitution_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        with self.assertRaisesRegex(launch.ContractViolation, "native launcher byte identity mismatch"):
            validate_request(request, contract_raw, b"hostile launcher", runtime_bytes)

    def test_external_launcher_pin_mismatch_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        with self.assertRaisesRegex(launch.ContractViolation, "external pin mismatch"):
            validate_request(
                request,
                contract_raw,
                launcher_raw,
                runtime_bytes,
                external_launcher_sha256="00" * 32,
            )

    def test_symlink_native_launcher_declaration_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        request["launcher"]["symlink_or_reparse"] = True  # type: ignore[index]
        with self.assertRaisesRegex(launch.ContractViolation, "symlink/reparse"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_generic_python_invocation_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        request["launcher"]["canonical_absolute_path"] = "/usr/bin/python"  # type: ignore[index]
        with self.assertRaisesRegex(launch.ContractViolation, "generic Python"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_ordinary_mutable_venv_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        request["runtime_image"]["ordinary_mutable_venv"] = True  # type: ignore[index]
        with self.assertRaisesRegex(launch.ContractViolation, "ordinary mutable venv"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_runtime_not_authenticated_before_startup_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        request["runtime_image"]["authenticated_before_python_startup"] = False  # type: ignore[index]
        with self.assertRaisesRegex(launch.ContractViolation, "before Python startup"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_mutable_runtime_image_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        request["runtime_image"]["platform_immutable"] = False  # type: ignore[index]
        with self.assertRaisesRegex(launch.ContractViolation, "platform immutable"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_hostile_pre_startup_encodings_substitution_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        runtime_bytes["lib/python3.12/encodings/__init__.py"] = b"hostile pre-startup code"
        with self.assertRaisesRegex(launch.ContractViolation, "encodings/__init__.py byte identity mismatch"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_missing_pre_startup_encodings_member_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        runtime_bytes.pop("lib/python3.12/encodings/__init__.py")
        request["runtime_image"]["members"] = [  # type: ignore[index]
            row
            for row in request["runtime_image"]["members"]  # type: ignore[index]
            if row["relative_path"] != "lib/python3.12/encodings/__init__.py"
        ]
        request["runtime_image"]["image_id"] = launch.runtime_image_id(  # type: ignore[index]
            request["runtime_image"]["members"]  # type: ignore[index]
        )
        with self.assertRaisesRegex(launch.ContractViolation, "missing authenticated pre-startup"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_runtime_member_symlink_declaration_rejected(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        request["runtime_image"]["members"][0]["symlink_or_reparse"] = True  # type: ignore[index]
        with self.assertRaisesRegex(launch.ContractViolation, "link/reparse"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)

    def test_pre_start_order_cannot_move_python_before_runtime_auth(self) -> None:
        request, contract_raw, launcher_raw, runtime_bytes = build_request()
        order = request["pre_start_order"]  # type: ignore[assignment]
        order[3], order[5] = order[5], order[3]
        with self.assertRaisesRegex(launch.ContractViolation, "pre-start order"):
            validate_request(request, contract_raw, launcher_raw, runtime_bytes)


class InertnessTests(unittest.TestCase):
    def test_bootstrap_source_default_is_terminally_inert(self) -> None:
        self.assertFalse(bootstrap_v4.AUTHORITY_GRANTED)
        self.assertEqual(bootstrap_v4.source_only_status()["status"], "FOSP_V4_SOURCE_ONLY_NATIVE_LAUNCHER_ABSENT")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(bootstrap_v4._main(), 78)
        self.assertIn("NATIVE_LAUNCHER_ABSENT", stream.getvalue())

    def test_contract_and_validator_authorize_nothing(self) -> None:
        contract, _, _ = validated_contract()
        self.assertFalse(launch.AUTHORITY_GRANTED)
        self.assertTrue(all(value is False for value in contract["authorization"].values()))
        self.assertTrue(all(value == 0 for value in contract["zero_access"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
