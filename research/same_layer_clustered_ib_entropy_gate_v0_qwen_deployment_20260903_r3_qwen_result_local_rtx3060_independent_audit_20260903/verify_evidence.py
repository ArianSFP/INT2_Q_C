"""Independent, stdlib-only audit of the completed local CBIB-r3 Qwen child result.

This verifier never imports NumPy or CuPy, opens no Qwen payload, invokes no GPU,
and cannot re-execute the consumed one-use authority.  It authenticates preserved
execution evidence and recomputes the reported bit-accounting/read conclusions.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import stat


VERDICT = (
    "PASS_COMPLETED_CHILD_RESULT_WITH_HARMLESS_STDERR_WARNING__"
    "HARD_KILL_CBIB_FIXED_LABEL"
)
RESULT_SHA256 = "e24d8795c655704732a42b2fb6e39ca323c2cc09d0d0e5cf34a070de9ef5b916"
DEPLOYMENT_MANIFEST_SHA256 = "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
DEPLOYMENT_ROOT_SHA256 = "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
CAPABILITY_MANIFEST_SHA256 = "1f99bde6bf48f29554639e746dcad5f7f808f9af628af0ff4a850a0e54ecd63b"
CAPABILITY_ROOT_SHA256 = "8f36ce5e3476e7058375622c2175b0624b27fb04904f4f1d70ac3baec3565fbf"
PANEL_SHA256 = "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5"
WEIGHTS = 50_331_648
EXPECTED_EXPERTS = list(range(0, 128, 8))
EXPECTED_SOURCE_FILES = {
    "CBIB1_R3_LOCAL3060_AUTHORITY_ATTEMPT_20260903_R3_6B20D57E.json":
        (441, "80074ccd1d92ead41fb9823e6a07fd313722b0b10e3279799ae1974f7c9a4ca2"),
    "CBIB1_R3_LOCAL3060_AUTHORITY_STATUS_20260903_R3_6B20D57E.json":
        (329, "bfaba514694476399b10cb668f034030feb3e654c52d6871d82dc8e2be1a4405"),
    "ONE_USE_CLAIM.json":
        (444, "d7382c6f42104606078fb36f086d97811e98833a372279393d70cfb1f787f15e"),
    "child_stderr.txt":
        (199, "70323dc64f419beb59b8ec0fc3af4ca002e09180df6ee51eb92967a033dbcd6e"),
    "child_stdout.jsonl":
        (414, "90915e254ed2c83eb92eac85744ddaa0505ced17241229f23b92a22bd59583eb"),
    "result.json": (217033, RESULT_SHA256),
}
EXPECTED_STDERR = (
    "C:\\INT2__compression\\.venv-cupy\\Lib\\site-packages\\cupy\\_environment.py:286: "
    "UserWarning: CUDA path could not be detected. Set CUDA_PATH environment variable "
    "if CuPy fails to load.\r\n  warnings.warn(\r\n"
).encode()


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, *, atol: float = 2e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def real_file(path: Path) -> None:
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
         f"regular nonsymlink file required: {path}")


def verify_frozen_manifest(root: Path, expected_manifest: str,
                           expected_root: str) -> dict:
    root = root.resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), "real frozen package root")
    manifest_path = root / "SOURCE_MANIFEST.json"
    real_file(manifest_path)
    need(sha(manifest_path) == expected_manifest, "frozen manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical frozen manifest")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "frozen manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)),
         "frozen manifest ordering")
    need(sorted(p.name for p in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "frozen package closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "frozen row fields")
        member = root / row["name"]
        real_file(member)
        need(member.stat().st_size == int(row["bytes"]) and sha(member) == row["sha256"],
             f"frozen member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed_root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(observed_root == expected_root == manifest.get("source_root_sha256"),
         "frozen package root digest")
    return manifest


def authenticate_execution_files(audit_root: Path) -> None:
    for name, (size, digest) in EXPECTED_SOURCE_FILES.items():
        member = audit_root / name
        real_file(member)
        need(member.stat().st_size == size and sha(member) == digest,
             f"preserved execution member mismatch: {name}")


def audit_claim_chain(audit_root: Path) -> tuple[dict, dict]:
    outer = json.loads((audit_root /
        "CBIB1_R3_LOCAL3060_AUTHORITY_ATTEMPT_20260903_R3_6B20D57E.json").read_bytes())
    status = json.loads((audit_root /
        "CBIB1_R3_LOCAL3060_AUTHORITY_STATUS_20260903_R3_6B20D57E.json").read_bytes())
    inner = json.loads((audit_root / "ONE_USE_CLAIM.json").read_bytes())
    need(outer.get("schema") == "cbib1-r3-local3060-authority-attempt-claim-v0-r3" and
         outer.get("status") == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS" and
         outer.get("capability_manifest_sha256") == CAPABILITY_MANIFEST_SHA256 and
         outer.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST_SHA256,
         "outer one-use claim")
    need(status == {
        "capability_manifest_sha256": CAPABILITY_MANIFEST_SHA256,
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "error_type": "RuntimeError",
        "schema": "cbib1-r3-local3060-authority-wrapper-v0-r3",
        "status": "FAIL_ATTEMPT_CONSUMED_NO_RETRY_AUTHORIZED",
    }, "outer wrapper status")
    need(inner.get("schema") == "same-layer-clustered-ib-one-use-claim-v0-r3" and
         inner.get("status") == "CONSUMED_BEFORE_PAYLOAD_ACCESS" and
         inner.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST_SHA256 and
         inner.get("parent_source_manifest_sha256") ==
             "1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a" and
         inner.get("parent_audit_manifest_sha256") ==
             "5c07e720928f2642867524b201d0abef5a17ea57b4cae68f5c0df59010e3f051",
         "inner one-use claim")
    return outer, inner


def audit_child_completion(audit_root: Path) -> dict:
    raw = (audit_root / "child_stdout.jsonl").read_bytes()
    need(raw.endswith(b"\n") and len(raw.splitlines()) == 1, "single child receipt")
    launch = json.loads(raw)
    need(set(launch) == {"claim", "output", "result_sha256", "schema", "status"} and
         launch["schema"] == "same-layer-clustered-ib-one-use-launch-receipt-v0-r3" and
         launch["status"] == "HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET" and
         launch["result_sha256"] == RESULT_SHA256 and
         launch["claim"].endswith("\\ONE_USE_CLAIM.json") and
         launch["output"].endswith("\\result.json"), "terminal child receipt")
    # The authenticated child writes this receipt only after result serialization and
    # immediately returns zero.  The outer wrapper's first subsequent content check is
    # an empty-stderr assertion.  The exact stderr is one CuPy warning and no traceback.
    stderr = (audit_root / "child_stderr.txt").read_bytes()
    need(stderr == EXPECTED_STDERR and b"Traceback" not in stderr and b"Error" not in stderr,
         "stderr is not the exact harmless CuPy warning")
    return launch


def audit_input_ledger(result: dict, panel: dict) -> None:
    ledger = result.get("input_read_ledger", {})
    rows = ledger.get("authenticated_inputs", [])
    expected = [
        {key: row[key] for key in ("bytes", "expert", "relative_path", "role", "sha256")}
        for row in panel["files"]
    ]
    need(rows == expected, "result inputs do not exactly match frozen panel")
    need(panel.get("model") == "Qwen/Qwen3-30B-A3B" and
         panel.get("revision") == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" and
         panel.get("layer") == 15 and panel.get("experts") == EXPECTED_EXPERTS,
         "panel identity")
    need(len(rows) == 32 and sum(int(row["bytes"]) for row in rows) == 100_663_296 and
         ledger.get("source_files_read_once") == 32 and
         ledger.get("source_bytes_read_once") == 100_663_296 and
         ledger.get("source_logical_host_scan_amplification") == 1.0 and
         ledger.get("scale_bits") == 393_216 and
         ledger.get("scale_bytes_per_expert") == 3072,
         "input read ledger")


def recompute_score(row: dict) -> None:
    weights = int(row["source_weights"])
    need(weights == WEIGHTS and int(row["expert_count"]) == 16 and
         int(row["roles"]) == 2 and int(row["coordinates_per_role"]) == 1_572_864,
         "score geometry")
    baseline_data = float(row["baseline_data_bits"])
    private_data = float(row["private_conditional_data_bits"])
    latent_data = float(row["latent_data_bits"])
    baseline_charged = (
        baseline_data + int(row["baseline_model_bits"]) + int(row["baseline_framing_bits"])
    )
    structured_charged = (
        private_data + latent_data + int(row["latent_model_bits"])
        + int(row["conditional_model_bits"]) + int(row["partition_bits"])
        + int(row["selector_bits"]) + int(row["structured_framing_bits"])
    )
    need(close(float(row["favorable_gross_gain_bpw"]),
               (baseline_data - private_data) / weights), "gross gain formula")
    need(close(float(row["net_ideal_gain_bpw"]),
               (baseline_data - private_data - latent_data) / weights),
         "net ideal formula")
    need(close(float(row["baseline_charged_bits"]), baseline_charged),
         "baseline charged bits formula")
    need(close(float(row["structured_charged_bits"]), structured_charged),
         "structured charged bits formula")
    need(close(float(row["charged_gain_bpw"]),
               (baseline_charged - structured_charged) / weights),
         "charged gain formula")
    need(len(row["private_data_bits_by_expert"]) == 16 and
         close(sum(map(float, row["private_data_bits_by_expert"])), private_data, atol=2e-7),
         "private-data sum")
    need(close(sum(map(float, row["common_data_bits_by_segment"])), latent_data,
               atol=2e-7), "common-data sum")
    need(sum(map(int, row["private_model_bits_by_expert"])) ==
         int(row["conditional_model_bits"]), "private-model sum")
    need(sum(map(int, row["common_model_bits_by_segment"])) ==
         int(row["latent_model_bits"]), "common-model sum")


def audit_read_envelope(row: dict, name: str) -> None:
    env = row["read_envelopes"][name]
    total_weights = int(row["source_weights"])
    actual = Fraction(int(env["total_pages"]) * 4096 * 8, total_weights)
    need(str(actual) == env["actual_rate_fraction"] and
         close(float(actual), float(env["actual_rate_bpw"])) and
         int(env["minimum_required_pages"]) <= int(env["total_pages"]),
         "read rate/capacity formula")
    physical = [Fraction(x) for x in env["amplification_physical_fraction"]]
    nonpadding = [Fraction(x) for x in env["amplification_nonpadding_fraction"]]
    need(len(physical) == len(nonpadding) == 16 and
         all(value < 2 for value in physical + nonpadding), "strict read bound")
    observed = max(physical + nonpadding)
    need(close(float(observed), float(env["max_amplification"])) and
         env["capacity_ok"] is True and env["strictly_below_2x"] is True and
         env["status"] == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC",
         "read-envelope status/max")
    for touched, ap, an, owned_p, owned_n in zip(
            env["touched_bytes"], physical, nonpadding,
            env["owned_physical_bytes"], env["owned_nonpadding_bytes"]):
        need(Fraction(int(touched), 1) / Fraction(owned_p) == ap and
             Fraction(int(touched), 1) / Fraction(owned_n) == an,
             "read amplification fraction")


def audit_result(audit_root: Path, panel: dict) -> dict:
    result_path = audit_root / "result.json"
    need(sha(result_path) == RESULT_SHA256, "result digest")
    result = json.loads(result_path.read_bytes())
    need(result.get("schema") == "same_layer_clustered_ib_entropy_gate_result_v0" and
         result.get("status") == "HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET" and
         result.get("eligible_for_finite_codec") is False and
         result.get("controls_executed") is True and
         result.get("claim_boundary") ==
             "IDEAL_LABEL_ENTROPY_CENSUS_ONLY_NOT_A_FINITE_CODEC_OR_MSE_RESULT" and
         result.get("panel_lock_sha256") == PANEL_SHA256 and
         close(float(result.get("target_gain_bpw_on_up_down")), 0.22933495044437174),
         "result identity/verdict")
    cuda = result.get("cuda", {})
    need(cuda == {"cupy_version": "14.2.0", "device_id": 0,
                  "device_name": "NVIDIA GeForce RTX 3060",
                  "driver_version": 12060, "runtime_version": 12090},
         "CUDA result identity")
    runtime = result.get("runtime_numpy_closure", {})
    closures = runtime.get("local_runtime_all_wheel_closures", [])
    need(runtime.get("version") == "2.5.2" and
         runtime.get("verified_before_one_use_claim") is True and
         runtime.get("unhashed_record_self_only") is True and
         runtime.get("device_uuid") == "GPU-458a424a-76e3-65e5-0470-803e0ed131ca" and
         runtime.get("cupy_uuid_canonical") == runtime.get("device_uuid") and
         runtime.get("cupy_uuid_raw_length") == 19 and
         runtime.get("cupy_uuid_trailing_hex") == "360701" and
         len(closures) == 10 and len({row["distribution"] for row in closures}) == 10,
         "runtime closure receipt")
    audit_input_ledger(result, panel)

    source = result.get("source_scores", [])
    need([int(row["group_size"]) for row in source] == [2, 4, 8, 16],
         "source group bank")
    for row in source:
        recompute_score(row)
        need(row.get("feasible_rate_endpoints") == ["43/20", "5/2"],
             "rate endpoint eligibility")
        audit_read_envelope(row, "43/20")
        audit_read_envelope(row, "5/2")

    favorable = [row for row in source
                 if float(row["favorable_gross_gain_bpw"]) >=
                    float(result["target_gain_bpw_on_up_down"])]
    need([int(row["group_size"]) for row in favorable] == [2],
         "only group-2 reaches favorable gross gate")
    controls = result.get("controls", [])
    need([int(control["seed"]) for control in controls] ==
         [3407544321, 3407544323, 3407544327, 3407544331,
          3407544337, 3407544339, 3407544343, 3407544349],
         "control seed bank")
    control_gains = []
    for control in controls:
        need(len(control["scores"]) == 1 and
             int(control["scores"][0]["group_size"]) == 2, "control candidate bank")
        recompute_score(control["scores"][0])
        control_gains.append(float(control["scores"][0]["charged_gain_bpw"]))
    group2 = source[0]
    maximum_control = max(control_gains)
    corrected = float(group2["charged_gain_bpw"]) - max(0.0, maximum_control)
    need(close(maximum_control, float(group2["maximum_control_charged_gain_bpw"])) and
         close(corrected, float(group2["control_corrected_charged_gain_bpw"])) and
         corrected < float(result["target_gain_bpw_on_up_down"]),
         "control correction and hard-kill")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-package", required=True, type=Path)
    parser.add_argument("--deployment-package", required=True, type=Path)
    parser.add_argument("--capability-package", required=True, type=Path)
    args = parser.parse_args()
    root = args.audit_package.resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), "real audit package")
    authenticate_execution_files(root)
    deployment = verify_frozen_manifest(
        args.deployment_package, DEPLOYMENT_MANIFEST_SHA256, DEPLOYMENT_ROOT_SHA256)
    capability = verify_frozen_manifest(
        args.capability_package, CAPABILITY_MANIFEST_SHA256, CAPABILITY_ROOT_SHA256)
    need(capability.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST_SHA256 and
         capability.get("deployment_source_root_sha256") == DEPLOYMENT_ROOT_SHA256 and
         capability.get("panel_lock_sha256") == PANEL_SHA256 and
         capability.get("status") == "AUTHORIZED_NOT_EXECUTED",
         "capability-to-deployment binding")
    need(deployment.get("parent_source_manifest_sha256") ==
             "1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a" and
         deployment.get("parent_audit_manifest_sha256") ==
             "5c07e720928f2642867524b201d0abef5a17ea57b4cae68f5c0df59010e3f051",
         "deployment lineage")
    audit_claim_chain(root)
    launch = audit_child_completion(root)
    panel_path = args.deployment_package.resolve(strict=True) / "panel_lock.json"
    need(sha(panel_path) == PANEL_SHA256, "panel digest")
    result = audit_result(root, json.loads(panel_path.read_bytes()))
    need(launch["status"] == result["status"], "child/result terminal status")
    group2 = result["source_scores"][0]
    print(json.dumps({
        "capability_manifest_sha256": CAPABILITY_MANIFEST_SHA256,
        "capability_source_root_sha256": CAPABILITY_ROOT_SHA256,
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "deployment_source_root_sha256": DEPLOYMENT_ROOT_SHA256,
        "group2_charged_gain_bpw": group2["charged_gain_bpw"],
        "group2_gross_private_gain_bpw": group2["favorable_gross_gain_bpw"],
        "group2_net_ideal_gain_bpw": group2["net_ideal_gain_bpw"],
        "result_sha256": RESULT_SHA256,
        "schema": "cbib-r3-qwen-result-independent-evidence-verification-v0",
        "status": VERDICT,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
