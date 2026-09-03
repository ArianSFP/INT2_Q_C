"""Read-only verifier for the consumed CBIB-1 local RTX3060 r1 attempt."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat


CAP_MANIFEST = "1589197b2ad053efacd7287b667294e6e8b4d736bff78196a1c29d991613fb5d"
CAP_ROOT = "28b639186be82851e409c3ec9058ddca01acc816be54ff58473628e325c0f166"
WRAPPER = "c388a7c8e4377f009c368e1a6857286aff525eca74d15c13d711071b05045ceb"
CLAIM = "e5c9cd1fd09a91dbe403178c65cea6bb9a1d87d212de08d05f19373a003c195f"
STATUS = "eafcf3e422e87da229a4bba3f24b41cde435516b8a17869ed33b1e9be0c97d2f"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path) -> None:
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--capability", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    capability = args.capability.resolve(strict=True)
    evidence_path = args.evidence.resolve(strict=True)
    regular(evidence_path)
    raw = evidence_path.read_bytes()
    evidence = json.loads(raw)
    need(raw == (json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "noncanonical evidence")
    need(evidence.get("schema") == "cbib1-r3-local3060-r1-failure-evidence-v0" and
         evidence.get("status") == "FAIL_ATTEMPT_CONSUMED_BEFORE_GPU_OR_QWEN_ACCESS",
         "evidence identity")

    manifest = capability / "SOURCE_MANIFEST.json"
    wrapper = capability / "run_authorized_local_qwen_once.py"
    regular(manifest); regular(wrapper)
    need(sha(manifest) == CAP_MANIFEST and sha(wrapper) == WRAPPER, "r1 source identity")
    manifest_obj = json.loads(manifest.read_bytes())
    need(manifest_obj.get("source_root_sha256") == CAP_ROOT, "r1 source root")
    source = wrapper.read_text(encoding="utf-8")
    ast.parse(source, filename=str(wrapper))
    ordered = ["exclusive_write(ATTEMPT_CLAIM, claim_raw)",
               "validate_prerequisites(args.capability_manifest_sha256)",
               "os.mkdir(RUN_ROOT)", "os.mkdir(CACHE_ROOT)",
               "os.open(str(CHILD_STDOUT)", "subprocess.run("]
    cursor = -1
    main_source = source[source.index("def main("):]
    for token in ordered:
        cursor = main_source.index(token, cursor + 1)
    need(not any(token in source for token in (".unlink(", "rmtree(", "os.remove(")),
         "unexpected cleanup path")

    paths = evidence["paths"]
    claim = workspace / paths["claim"]
    status = workspace / paths["status"]
    run_root = workspace / paths["run_root"]
    cache = workspace / paths["cache_root"]
    result = workspace / paths["result"]
    regular(claim); regular(status)
    need(sha(claim) == CLAIM and sha(status) == STATUS, "retained artifact digest")
    claim_obj = json.loads(claim.read_bytes())
    status_obj = json.loads(status.read_bytes())
    need(claim_obj.get("capability_manifest_sha256") == CAP_MANIFEST and
         claim_obj.get("status") == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS" and
         status_obj.get("status") == "FAIL_ATTEMPT_CONSUMED_NO_RETRY_AUTHORIZED" and
         status_obj.get("error_type") == "PermissionError", "retained semantics")
    need(run_root.is_dir() and not run_root.is_symlink() and not any(run_root.iterdir()),
         "run root not empty")
    need(not cache.exists() and not cache.is_symlink() and
         not result.exists() and not result.is_symlink(), "cache or result exists")
    need(not (run_root / "child_stdout.jsonl").exists() and
         not (run_root / "child_stderr.txt").exists() and
         not (run_root / "ONE_USE_CLAIM.json").exists(), "child artifact exists")
    need(evidence["inference"] == {"child_bridge_launched": False,
         "failed_operation": "os.mkdir(CACHE_ROOT)", "gpu_initialized": False,
         "qwen_payload_opened": False, "result_created": False}, "inference mismatch")
    print(json.dumps({"attempt_consumed": True, "failed_operation": "os.mkdir(CACHE_ROOT)",
        "gpu_initialized": False, "qwen_payload_opened": False, "result_created": False,
        "schema": "cbib1-r3-local3060-r1-failure-verification-v0",
        "status": "PASS_SEALED_FAILURE_NO_GPU_QWEN_OR_RESULT_ACCESS"},
        sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
