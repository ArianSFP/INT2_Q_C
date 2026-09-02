#!/usr/bin/env python3
"""v1 authentication: open and hash every manifest named by the binding."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


V0_MANIFEST_SHA256 = "287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde"
V0_AUTH_SHA256 = "4d461bcdacb1a38fe129b2f320b82cb9d22cb57626b4413228dd197b67348382"


class AuthenticationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthenticationError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_v0_auth() -> Any:
    root = Path(__file__).resolve().parents[1] / "tactic_ramanujan384_adapter_v0"
    manifest = root / "SOURCE_MANIFEST.json"
    module_path = root / "authenticated_io.py"
    require(manifest.is_file() and module_path.is_file(), "pinned v0 authentication dependency")
    require(sha256(manifest.read_bytes()) == V0_MANIFEST_SHA256, "pinned v0 manifest drift")
    require(sha256(module_path.read_bytes()) == V0_AUTH_SHA256, "pinned v0 authentication drift")
    name = "tactic_ramanujan384_authority_v1_pinned_v0_auth"
    spec = importlib.util.spec_from_file_location(name, module_path)
    require(spec is not None and spec.loader is not None, "v0 authentication loader")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def authenticate_role(
    *,
    binding_path: Path,
    expected_binding_sha256: str,
    audit_receipt_path: Path,
    coarse_artifact_path: Path,
    source_bf16_path: Path,
    coarse_reconstruction_f32_path: Path,
    input_manifest_path: Path,
    auditor_source_manifest_path: Path,
) -> dict[str, Any]:
    """Authenticate literal inputs plus the actual two manifests, not labels."""

    v0 = _load_v0_auth()
    base = v0.authenticate_role(
        binding_path=binding_path,
        expected_binding_sha256=expected_binding_sha256,
        audit_receipt_path=audit_receipt_path,
        coarse_artifact_path=coarse_artifact_path,
        source_bf16_path=source_bf16_path,
        coarse_reconstruction_f32_path=coarse_reconstruction_f32_path,
    )
    binding_payload = v0.read_regular(binding_path, 1 << 20)
    binding = v0.strict_json(binding_payload, "binding")
    audit_payload = v0.read_regular(audit_receipt_path, 4 << 20)
    audit = v0.strict_json(audit_payload, "independent audit")
    input_payload = v0.read_regular(input_manifest_path, 16 << 20)
    auditor_payload = v0.read_regular(auditor_source_manifest_path, 16 << 20)
    input_digest = sha256(input_payload)
    auditor_digest = sha256(auditor_payload)
    require(input_digest == binding["input_manifest_sha256"],
            "actual input manifest SHA256")
    require(input_digest == audit.get("input_manifest_sha256"),
            "audit actual input manifest SHA256")
    require(auditor_digest == binding["independent_audit"]["auditor_source_manifest_sha256"],
            "actual auditor source manifest SHA256")
    require(auditor_digest == audit.get("auditor_source_manifest_sha256"),
            "audit actual auditor source manifest SHA256")
    # Parsing is mandatory so duplicate keys and non-finite constants cannot
    # hide behind a hash-only opaque file.  Schema ownership remains with the
    # independently pinned producer/auditor packages.
    v0.strict_json(input_payload, "actual input manifest")
    v0.strict_json(auditor_payload, "actual auditor source manifest")
    base.update({
        "actual_input_manifest_sha256": input_digest,
        "actual_auditor_source_manifest_sha256": auditor_digest,
        "actual_input_manifest_opened": True,
        "actual_auditor_source_manifest_opened": True,
        "manifest_paths_are_explicit_capabilities": True,
    })
    return base
