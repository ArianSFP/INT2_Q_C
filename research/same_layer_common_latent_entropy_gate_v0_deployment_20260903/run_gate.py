"""Fail-closed entrypoint for the Qwen layer-15 common-latent aperture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


# Source release is intentionally incapable of touching payload.  A separately
# reviewed execution copy must flip this literal after verifying SOURCE_MANIFEST.
PAYLOAD_EXECUTION_ENABLED = True
AUTHORIZATION_PHRASE = "EXECUTE_AUTHENTICATED_QWEN_L15_COMMON_LATENT_V0"
PANEL_LOCK_SHA256 = "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5"
CORE_SHA256 = "2ac6ce05dc4c2bc72d71acab443e92ae94917300150cd1d8fb5c8264daff04ea"
WORKER_SHA256 = "b238d57a8e1435556b9126577e6bede4bb9d3ddaad709a4e2adb54e3185dfeac"


def _hold(reason: str) -> int:
    print(json.dumps({
        "schema": "same_layer_common_latent_hold_v0",
        "status": "HOLD_NO_PAYLOAD_ACCESS",
        "reason": reason,
        "payload_execution_enabled": PAYLOAD_EXECUTION_ENABLED,
    }, sort_keys=True))
    return 2


def _load_verified_module(module_name: str, path: Path, expected_sha256: str):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError(f"source hash mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verified source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    # The worker imports common_latent_core by its frozen name.  Replace, rather
    # than trust, any preloaded module with the verified local snapshot.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", default="HOLD")
    parser.add_argument("--payload-root", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    # This branch precedes path resolution, file existence checks, NumPy/CuPy
    # import, output creation, and panel parsing.
    if not PAYLOAD_EXECUTION_ENABLED:
        return _hold("compile_time_payload_switch_is_false")
    if args.authorization != AUTHORIZATION_PHRASE:
        return _hold("authorization_phrase_mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")

    package = Path(__file__).resolve().parent
    panel_path = package / "panel_lock.json"
    panel_bytes = panel_path.read_bytes()
    if hashlib.sha256(panel_bytes).hexdigest() != PANEL_LOCK_SHA256:
        raise RuntimeError("panel lock source hash mismatch")
    payload_root = Path(args.payload_root)
    output = Path(args.output)
    if not payload_root.is_absolute() or not output.is_absolute():
        raise RuntimeError("payload root and output must be absolute paths")
    if not args.payload_root or not payload_root.is_dir() or payload_root.is_symlink():
        raise RuntimeError("payload root must be an explicit real directory")
    if not args.output or output.suffix.lower() != ".json" or output.exists():
        raise RuntimeError("output must be an explicit absent .json file")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise RuntimeError("output parent must be an existing real directory")

    # Imported only beyond the fail-closed payload boundary, and only from
    # byte-pinned source snapshots.
    _load_verified_module(
        "common_latent_core", package / "common_latent_core.py", CORE_SHA256
    )
    worker = _load_verified_module(
        "same_layer_common_latent_cupy_worker_v0",
        package / "cupy_worker.py",
        WORKER_SHA256,
    )
    result = worker.run_authorized_panel(panel_path, payload_root)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "result_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
