"""Hostile stdlib audit, with optional CPU reproduction, for the exact r3 package."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat
import sys


R3_MANIFEST = "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
R3_ROOT = "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
R2_MANIFEST = "d3e0fb929fe422bf55351470d445d683bbae7dd5d1bfcd50fbe3fe4a22530f0c"
R2_ROOT = "e2476a8ce54183ab5b5eb2a6f44bbffb21d39331d38de47d3f448d711a82699b"
R2_REVIEW_MANIFEST = "3522a9700e9344f3be7282291d8b994087fc235163dbb9ecc71f0d5e7e83e837"
R2_REVIEW_ROOT = "272b988f2e1390e5b3129b45db919feb268fe34d0ec7464cb91aa436da29df60"
SOURCE_MANIFEST = "1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a"
SOURCE_ROOT = "18a4043e99b17cfa535f4a6c2930f2c1ac42eff092f4e5d61b9408b1986f457e"
SOURCE_AUDIT_MANIFEST = "5c07e720928f2642867524b201d0abef5a17ea57b4cae68f5c0df59010e3f051"
SOURCE_AUDIT_ROOT = "2d0b25666b2dc20feef8dfa56fd62c377b7ba7e1c66e34c3844fb5d1b02b45ca"
CORE_SHA = "25e84b9d5e598a72984e48cb5593c41725d096e36082b20b3d47a78f2100e340"
WORKER_SHA = "a34ca17dd8f76afa0331bb56d5b5dec26dcde693d05755ea2ca342a76a6badfc"
PANEL_SHA = "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5"
PRIOR_PANEL_SHA = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
FIXTURE_SHA = "33f7ba9d4ae0589d06abcfab06bac46d06ef75188d714350ba993df0ca9bbab5"
EVIDENCE_SHA = "5b0f0fe567db43fe14ecf53c1f883945a95d71b886aee81d1f0510e1b134ae84"
STATUS = "PASS_R3_AUTHORIZE_ONE_SOURCE_FREE_RTX5090_PREFLIGHT_ONLY"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate(root: Path, manifest_sha: str, root_sha: str, schema: str) -> dict:
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), "real package root")
    manifest_path = root / "SOURCE_MANIFEST.json"
    need(sha(manifest_path) == manifest_sha, "manifest digest")
    raw = manifest_path.read_bytes()
    obj = json.loads(raw)
    need(raw == (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical manifest")
    need(obj.get("schema") == schema, "manifest schema")
    rows = obj.get("files")
    need(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "member order")
    need(sorted(path.name for path in root.iterdir()) == sorted(names + ["SOURCE_MANIFEST.json"]),
         "package closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "manifest row")
        member = root / row["name"]
        need(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink(), "member type")
        need(member.stat().st_size == int(row["bytes"]) and sha(member) == row["sha256"],
             f"member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(observed == root_sha == obj.get("source_root_sha256"), "source root")
    return obj


def literals(path: Path) -> dict:
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return result


def imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def ordered(text: str, snippets: tuple[str, ...]) -> None:
    cursor = -1
    for snippet in snippets:
        cursor = text.index(snippet, cursor + 1)


def cpu_reproduction(r3: Path) -> None:
    import numpy as np

    sys.path.insert(0, str(r3))
    import clustered_ib_core as core
    import run_source_free_cupy as preflight
    import source_free_fixture as fixture

    need(fixture.EXPERT_COUNT == 16 and fixture.ROLES == 2 and
         fixture.COORDINATES == 131072 and fixture.FOLD_COUNT == 8 and
         fixture.SUPERBLOCK_VALUES == 2048 and fixture.SCALE_BYTES_PER_VALUE == 2 and
         fixture.SCALE_BYTES_PER_EXPERT == 256, "fixture constants")
    need(fixture.SCALE_BYTES_PER_EXPERT ==
         fixture.ROLES * (fixture.COORDINATES // fixture.SUPERBLOCK_VALUES) *
         fixture.SCALE_BYTES_PER_VALUE, "scale formula")
    labels = fixture.make_production_geometry_survivor_fixture()
    need(labels.shape == (16, 2, 131072) and labels.dtype == np.uint8 and
         int(labels.min()) == 0 and int(labels.max()) == 3 and
         hashlib.sha256(labels.tobytes(order="C")).hexdigest() == FIXTURE_SHA,
         "fixture bytes")
    need(all(np.array_equal(labels[index, 1], labels[index ^ 1, 0])
             for index in range(16)), "role reflection")
    folds = core.fold_ids(131072, 8, 2048)
    need(np.bincount(folds, minlength=8).astype(int).tolist() == [16384] * 8,
         "fold coverage")
    score = core.crossfit_group_size(labels, 2, fold_count=8, superblock_values=2048)
    requirements = core.packet_requirements(score, 256)
    envelopes = {
        str(rate): core.physical_read_envelope(
            expert_count=16, weights_per_expert=262144,
            requested_rate=rate, **requirements,
        )
        for rate in core.RATE_ENDPOINTS
    }
    need(float(score["favorable_gross_gain_bpw"]) == 0.6513992144263967 and
         float(score["charged_gain_bpw"]) == 0.10902764891263406,
         "targeted gains")
    need(envelopes["43/20"]["status"] == "FAIL_PACKET_EXCEEDS_RATE_CAP" and
         envelopes["43/20"]["capacity_ok"] is False and
         envelopes["5/2"]["status"] == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC" and
         envelopes["5/2"]["capacity_ok"] is True and
         envelopes["5/2"]["strictly_below_2x"] is True and
         envelopes["5/2"]["total_pages"] == 320 and
         envelopes["5/2"]["minimum_required_pages"] == 306 and
         float(envelopes["5/2"]["max_amplification"]) == 1.9651249492746525,
         "targeted strict-read survivor")
    rng = np.random.default_rng(0xA3B)
    for group_size in (2, 4, 8, 16):
        for coordinates in (1, 2, 7, 64, 257):
            q = rng.integers(0, 4, size=(group_size, coordinates), dtype=np.uint8)
            assignment = rng.integers(0, 2, size=coordinates, dtype=np.uint8)
            vector = preflight._independent_counts(q, assignment, np)
            fallback = preflight._independent_counts(q.tolist(), assignment.tolist())
            frozen = tuple(value.tolist() for value in core._model_counts(q, assignment))
            need(vector == fallback == frozen, "independent count probe")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--prior-panel-manifest", required=True)
    parser.add_argument("--audit-directory", required=True)
    parser.add_argument("--run-targeted-cpu", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repository_root).resolve(strict=True)
    audit = Path(args.audit_directory).resolve(strict=True)
    prior_path = Path(args.prior_panel_manifest).resolve(strict=True)
    research = repo / "research"
    r3 = research / "same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3"
    r2 = research / "same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r2"
    r2_review = research / "same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r2_independent_review"
    source = research / "same_layer_clustered_ib_entropy_gate_v0"
    source_audit = research / "same_layer_clustered_ib_entropy_gate_v0_independent_source_audit_20260903"

    manifest = authenticate(r3, R3_MANIFEST, R3_ROOT,
                            "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3")
    authenticate(r2, R2_MANIFEST, R2_ROOT,
                 "same-layer-clustered-ib-qwen-deployment-manifest-v0-r2")
    authenticate(r2_review, R2_REVIEW_MANIFEST, R2_REVIEW_ROOT,
                 "same-layer-clustered-ib-qwen-deployment-r2-independent-review-manifest-v0")
    authenticate(source, SOURCE_MANIFEST, SOURCE_ROOT,
                 "same_layer_clustered_ib_source_manifest_v0")
    authenticate(source_audit, SOURCE_AUDIT_MANIFEST, SOURCE_AUDIT_ROOT,
                 "same-layer-clustered-ib-independent-source-audit-manifest-v0")
    need(len(manifest["files"]) == 15 and
         manifest["parent_source_manifest_sha256"] == SOURCE_MANIFEST and
         manifest["parent_source_root_sha256"] == SOURCE_ROOT and
         manifest["parent_audit_manifest_sha256"] == SOURCE_AUDIT_MANIFEST and
         manifest["parent_audit_root_sha256"] == SOURCE_AUDIT_ROOT and
         manifest["r2_deployment_manifest_sha256"] == R2_MANIFEST and
         manifest["r2_deployment_root_sha256"] == R2_ROOT and
         manifest["r2_review_manifest_sha256"] == R2_REVIEW_MANIFEST and
         manifest["r2_review_root_sha256"] == R2_REVIEW_ROOT, "lineage pins")
    for name, expected in (("clustered_ib_core.py", CORE_SHA),
                           ("cupy_worker.py", WORKER_SHA),
                           ("panel_lock.json", PANEL_SHA)):
        need(sha(r3 / name) == expected and (r3 / name).read_bytes() == (r2 / name).read_bytes(),
             f"byte exact to r2: {name}")

    need(sha(prior_path) == PRIOR_PANEL_SHA, "prior panel manifest")
    prior = json.loads(prior_path.read_bytes())
    panel = json.loads((r3 / "panel_lock.json").read_bytes())
    need(panel["model"] == prior["repository"] == "Qwen/Qwen3-30B-A3B" and
         panel["revision"] == prior["revision"] ==
         "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" and
         panel["layer"] == prior["layer"] == 15 and
         panel["experts"] == prior["experts_in_order"] == list(range(0, 128, 8)),
         "panel identity")
    prior_map = {(int(row["expert"]), str(row["role"])): row for row in prior["tensors"]}
    need(len(panel["files"]) == len(prior_map) == 32 and
         sum(int(row["bytes"]) for row in panel["files"]) == 100663296, "panel extent")
    for row in panel["files"]:
        ref = prior_map[(int(row["expert"]), str(row["role"]))]
        need(row["relative_path"] == ref["local_path"] and
             all(row[key] == ref[key] for key in
                 ("bytes", "sha256", "raw_shape", "canonical_shape", "down_transposed")),
             "panel projection")

    for production_name in ("clustered_ib_core.py", "cupy_worker.py", "run_gate.py"):
        need("source_free_fixture" not in imports(r3 / production_name) and
             "make_production_geometry_survivor_fixture" not in
             (r3 / production_name).read_text(encoding="utf-8"), "fixture leakage")
    worker_text = (r3 / "cupy_worker.py").read_text(encoding="utf-8")
    need('labels, int(io["scale_bytes_per_expert"])' in worker_text and
         "import clustered_ib_core as core" in worker_text and
         "common_latent_core" not in worker_text, "production worker provenance/scale")
    core_text = (r3 / "clustered_ib_core.py").read_text(encoding="utf-8")
    need("Qwen/" not in core_text and "qwen_weight_cache" not in core_text,
         "universal core")

    fixture_text = (r3 / "source_free_fixture.py").read_text(encoding="utf-8")
    for token in ("EXPERT_COUNT = 16", "ROLES = 2", "COORDINATES = 131_072",
                  "FOLD_COUNT = 8", "SUPERBLOCK_VALUES = 2_048",
                  "SCALE_BYTES_PER_VALUE = 2",
                  "SCALE_BYTES_PER_EXPERT = ROLES * BLOCKS_PER_ROLE * SCALE_BYTES_PER_VALUE",
                  "LATENT_PROBABILITY = 0.5", "SIGN_FLIP_PROBABILITY = 0.105"):
        need(token in fixture_text, f"fixture token: {token}")

    parity = (r3 / "run_source_free_cupy.py").read_text(encoding="utf-8")
    for token in ("def _independent_counts(", "cpu_train_eval = core.evaluate_binary_model(",
                  '"training", np', '"held-out", np',
                  "for group_size in (2, 4, 8, 16):", "for fold in range(FOLD_COUNT):",
                  "for role in range(2):", 'gpu_eval["test_latent_counts"]',
                  'gpu_eval["test_conditional_counts"]',
                  "labels, SCALE_BYTES_PER_EXPERT, fold_count=FOLD_COUNT",
                  "q_gpu, SCALE_BYTES_PER_EXPERT, fold_count=FOLD_COUNT",
                  '_compare(cpu_gate, gpu_gate, "gate", stats)',
                  'len(core.CONTROL_SEEDS) != 8'):
        need(token in parity, f"parity token: {token}")
    need('cpu_eval["test_latent_counts"]' not in parity and
         'cpu_eval["test_conditional_counts"]' not in parity, "r1 schema defect absent")
    ordered(parity, ("source_survivors = [", "if not source_survivors:",
                     'if cpu_gate["controls_executed"] is not True:',
                     'if len(cpu_gate["controls"])'))

    run_path = r3 / "run_gate.py"
    run = literals(run_path)
    run_text = run_path.read_text(encoding="utf-8")
    main_text = run_text[run_text.index("def main("):]
    need(run["AUTHORIZATION_PHRASE"] ==
         "EXECUTE_AUTHENTICATED_QWEN_L15_CBIB1_V0_R3_ONCE" and
         run["PAYLOAD_ROOT"] == "/workspace/INT2__compression" and
         run["OUTPUT_PARENT"] == "/tmp/codex_cbib1_qwen_l15_oneuse_20260903_r3" and
         "--payload-root" not in run_text and "--output" not in run_text, "fixed production capability")
    ordered(main_text, ("if args.authorization != AUTHORIZATION_PHRASE:",
                        "_verify_manifest(package, args.deployment_manifest_sha256)",
                        "_cp, numpy_receipt = _validate_runtime()", "payload_root = Path(PAYLOAD_ROOT)",
                        "if output.exists() or output.is_symlink():",
                        "if claim.exists() or claim.is_symlink():",
                        "_claim_once(claim, args.deployment_manifest_sha256)",
                        '_load_verified_module("clustered_ib_core"',
                        "result = worker.run_authorized_panel(panel_path, payload_root)",
                        'with output.open("x"'))
    need("os.O_WRONLY | os.O_CREAT | os.O_EXCL" in run_text and
         "claim_path.unlink" not in run_text, "production one-use")
    numpy_text = run_text[run_text.index("def _verify_numpy_record_closure"):
                          run_text.index("def _validate_runtime")]
    for token in ("np.__version__", "NUMPY_FILE_SHA256", "NUMPY_RECORD_SHA256",
                  'relative.startswith(("numpy/", "numpy.libs/", "numpy-2.5.2.dist-info/"))',
                  "hashlib.sha256(data).digest()", 'unhashed != ["numpy-2.5.2.dist-info/RECORD"]',
                  "rows_checked < 500", "native_checked < 10"):
        need(token in numpy_text, f"NumPy closure token: {token}")

    wrapper_text = (audit / "run_authorized_preflight_once.py").read_text(encoding="utf-8")
    ordered(wrapper_text[wrapper_text.index("def main("):],
            ("_exclusive_write(CLAIM_PATH, claim)", "_authenticate_package()",
             "receipt_fd = os.open(", "subprocess.run("))
    need("os.O_WRONLY | os.O_CREAT | os.O_EXCL" in wrapper_text and
         "shell=False" in wrapper_text and "run_source_free_cupy.py" in wrapper_text and
         "run_gate.py" not in wrapper_text and "CLAIM_PATH.unlink" not in wrapper_text,
         "sealed preflight wrapper")
    authority = json.loads((audit / "AUTHORIZED_PREFLIGHT.json").read_bytes())
    need(authority["permitted_attempts"] == 1 and authority["command"]["shell"] is False and
         authority["forbidden"]["authorized_qwen_invocations"] == 0 and
         authority["forbidden"]["authorized_payload_file_reads"] == 0 and
         authority["forbidden"]["authorized_capability_or_production_launcher_invocations"] == 0,
         "narrow preflight authority")
    need(sha(audit / "CPU_TARGETED_EVIDENCE.json") == EVIDENCE_SHA, "CPU evidence digest")
    evidence = json.loads((audit / "CPU_TARGETED_EVIDENCE.json").read_bytes())
    need(evidence["fixture_probe"]["scale_bytes_per_expert"] == 256 and
         evidence["sealed_targeted_regression_reproduction"]["feasible_rate_endpoints"] == ["5/2"] and
         evidence["sealed_targeted_regression_reproduction"]["read_envelopes"]["5/2"]
         ["max_amplification"] == 1.9651249492746525, "CPU evidence")
    receipt = json.loads((audit / "AUDIT_RECEIPT.json").read_bytes())
    need(receipt["status"] == STATUS and
         receipt["audited_deployment"]["manifest_sha256"] == R3_MANIFEST and
         receipt["audited_deployment"]["source_root_sha256"] == R3_ROOT and
         receipt["authorization"]["authorized_source_free_rtx5090_preflight_attempts"] == 1 and
         receipt["authorization"]["authorized_qwen_invocations"] == 0 and
         receipt["authorization"]["authorized_capability_or_production_launcher_invocations"] == 0,
         "PASS receipt")
    if args.run_targeted_cpu:
        cpu_reproduction(r3)
    print(json.dumps({
        "authorized_capability_or_production_launcher_invocations": 0,
        "authorized_qwen_invocations": 0,
        "authorized_source_free_rtx5090_preflight_attempts": 1,
        "cpu_reproduced": bool(args.run_targeted_cpu),
        "deployment_manifest_sha256": R3_MANIFEST,
        "deployment_source_root_sha256": R3_ROOT,
        "payload_accessed": False,
        "schema": "same-layer-clustered-ib-qwen-deployment-r3-independent-audit-reproduction-v0",
        "status": STATUS,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
