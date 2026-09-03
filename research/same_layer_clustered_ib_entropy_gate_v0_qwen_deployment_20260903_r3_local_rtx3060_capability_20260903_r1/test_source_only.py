"""Hostile source-only tests.  Never exercise the production authorization path."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "run_authorized_local_qwen_once.py"
BRIDGE = ROOT / "local_runtime_bridge.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    bridge_text = BRIDGE.read_text(encoding="utf-8")
    ast.parse(wrapper_text, filename=str(WRAPPER))
    ast.parse(bridge_text, filename=str(BRIDGE))
    ordered = [
        "if args.authorization != AUTHORIZATION:",
        "exclusive_write(ATTEMPT_CLAIM, claim_raw)",
        "validate_prerequisites(args.capability_manifest_sha256)",
        "subprocess.run(",
        "validate_result()",
    ]
    cursor = -1
    main_text = wrapper_text[wrapper_text.index("def main("):]
    for token in ordered:
        cursor = main_text.index(token, cursor + 1)
    assert "shell=False" in wrapper_text
    assert "os.O_CREAT | os.O_EXCL" in wrapper_text
    assert not any(token in wrapper_text for token in (".unlink(", "rmtree(", "remove("))
    assert "os.add_dll_directory" in bridge_text
    assert "CUPY_CACHE_DIR" in bridge_text and "CUDA_PATH" in bridge_text
    assert "module._validate_runtime = validate_local_runtime" in bridge_text
    assert "module.clustered_ib_core" not in bridge_text and "module.cupy_worker" not in bridge_text
    assert not any(token in wrapper_text + bridge_text for token in
                   ("urllib", "requests", "socket", "http://", "https://"))

    wrapper = load("cbib1_local_wrapper_source_test", WRAPPER)
    bridge = load("cbib1_local_bridge_source_test", BRIDGE)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        assert wrapper.main(["--authorization", "HOLD"]) == 2
        assert bridge.main(["--authorization", "HOLD"]) == 2
    holds = [json.loads(line) for line in capture.getvalue().splitlines()]
    assert all("HOLD" in row["status"] for row in holds)

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        claim = temporary_path / "claim.json"
        status = temporary_path / "status.json"
        old_cwd = Path.cwd()
        os.chdir(ROOT)
        try:
            with mock.patch.object(wrapper, "ATTEMPT_CLAIM", claim), \
                    mock.patch.object(wrapper, "AUTHORITY_STATUS", status), \
                    mock.patch.object(wrapper, "AUTHORITY_ROOT", ROOT), \
                    mock.patch.object(wrapper, "validate_prerequisites",
                                      side_effect=RuntimeError("source-only sentinel")), \
                    mock.patch.object(wrapper.subprocess, "run") as child:
                try:
                    wrapper.main(["--authorization", wrapper.AUTHORIZATION,
                                  "--capability-manifest-sha256", "0" * 64])
                except RuntimeError as exc:
                    assert str(exc) == "source-only sentinel"
                else:
                    raise AssertionError("sentinel validation failure was not propagated")
                assert claim.is_file() and status.is_file() and not child.called
                assert json.loads(claim.read_bytes())["status"] == \
                    "ATTEMPT_CONSUMED_BEFORE_VALIDATION_OR_PAYLOAD_ACCESS"
                try:
                    wrapper.main(["--authorization", wrapper.AUTHORIZATION,
                                  "--capability-manifest-sha256", "0" * 64])
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("second invocation did not fail at O_EXCL claim")
        finally:
            os.chdir(old_cwd)

    assert "numpy" not in sys.modules and "cupy" not in sys.modules
    print(json.dumps({
        "ast_passed": True,
        "bridge_hold_passed": True,
        "claim_precedes_fallible_validation": True,
        "gpu_initialized": False,
        "network_accessed": False,
        "payload_accessed": False,
        "production_executed": False,
        "schema": "cbib1-r3-local3060-capability-source-only-test-v0-r1",
        "second_attempt_rejected": True,
        "shell_free_child": True,
        "status": "PASS_SOURCE_ONLY_NOT_EXECUTED",
        "wrapper_hold_passed": True,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
