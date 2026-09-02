#!/usr/bin/env python3
"""Object-authenticating production boundary for the source-only v2 gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any

from codec import CodecError


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_SCHEMA = "strata-identical-selection-gaussian-control-receipt-v1"
READ_SCHEMA = "strata-routed-cold-read-ledger-v1"
AUDIT_SCHEMA = "strata-bmp-qtt6-independent-audit-receipt-v2"


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise CodecError(f"{label} duplicate JSON key")
            value[key] = item
        return value
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CodecError(f"{label} nonfinite JSON {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodecError(f"{label} strict JSON") from exc
    if not isinstance(value, dict):
        raise CodecError(f"{label} JSON object")
    return value


@dataclass(frozen=True)
class ArtifactBinding:
    """A literal regular object plus its externally supplied SHA-256."""

    path: Path
    expected_sha256: str

    def authenticate(self, label: str) -> tuple[bytes, dict]:
        if not isinstance(self.path, Path):
            raise CodecError(f"{label} path object")
        if not isinstance(self.expected_sha256, str) or not SHA256_RE.fullmatch(
                self.expected_sha256):
            raise CodecError(f"{label} external SHA-256")
        try:
            before = self.path.lstat()
            if self.path.is_symlink() or not stat.S_ISREG(before.st_mode):
                raise CodecError(f"{label} regular non-link")
            payload = self.path.read_bytes()
            after = self.path.lstat()
        except OSError as exc:
            raise CodecError(f"{label} object read") from exc
        identity_before = (before.st_size, before.st_mtime_ns, before.st_mode,
                           before.st_ino)
        identity_after = (after.st_size, after.st_mtime_ns, after.st_mode,
                          after.st_ino)
        if identity_before != identity_after:
            raise CodecError(f"{label} changed during read")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.expected_sha256:
            raise CodecError(f"{label} object SHA-256")
        return payload, {"sha256": digest, "bytes": len(payload)}


@dataclass(frozen=True)
class ProductionHooks:
    strata_packet: ArtifactBinding | None = None
    scale_decoder: ArtifactBinding | None = None
    forward_transform: ArtifactBinding | None = None
    inverse_transform: ArtifactBinding | None = None
    original_bf16_scorer: ArtifactBinding | None = None
    gaussian_control_factory: ArtifactBinding | None = None
    component_framer: ArtifactBinding | None = None
    routed_read_ledger: ArtifactBinding | None = None
    independent_audit_receipt: ArtifactBinding | None = None
    gaussian_control_receipts: tuple[ArtifactBinding, ...] = ()
    expected_source_manifest_sha256: str | None = None
    expected_source_root_sha256: str | None = None

    def authorize(self) -> dict:
        fixed = {
            "strata_packet": self.strata_packet,
            "scale_decoder": self.scale_decoder,
            "forward_transform": self.forward_transform,
            "inverse_transform": self.inverse_transform,
            "original_bf16_scorer": self.original_bf16_scorer,
            "gaussian_control_factory": self.gaussian_control_factory,
            "component_framer": self.component_framer,
            "routed_read_ledger": self.routed_read_ledger,
            "independent_audit_receipt": self.independent_audit_receipt,
        }
        missing = [name for name, binding in fixed.items()
                   if not isinstance(binding, ArtifactBinding)]
        if missing:
            raise CodecError("unbound production objects: " + ",".join(missing))
        if len(self.gaussian_control_receipts) < 8 or not all(
                isinstance(row, ArtifactBinding)
                for row in self.gaussian_control_receipts):
            raise CodecError("at least eight authenticated Gaussian controls")
        if (not isinstance(self.expected_source_manifest_sha256, str) or
                not SHA256_RE.fullmatch(self.expected_source_manifest_sha256) or
                not isinstance(self.expected_source_root_sha256, str) or
                not SHA256_RE.fullmatch(self.expected_source_root_sha256)):
            raise CodecError("external source manifest/root pins")

        authenticated = {}
        payloads = {}
        for name, binding in fixed.items():
            assert isinstance(binding, ArtifactBinding)
            payload, receipt = binding.authenticate(name)
            payloads[name] = payload
            authenticated[name] = receipt

        controls = []
        control_ids = set()
        control_digests = set()
        for index, binding in enumerate(self.gaussian_control_receipts):
            payload, receipt = binding.authenticate(f"gaussian_control_{index}")
            value = _strict_json(payload, f"gaussian_control_{index}")
            if (value.get("schema") != CONTROL_SCHEMA or
                    value.get("identical_complete_selection_replayed") is not True or
                    value.get("selected_control") is not True or
                    not isinstance(value.get("control_id"), str) or
                    not value["control_id"]):
                raise CodecError(f"gaussian_control_{index} semantic receipt")
            if (value["control_id"] in control_ids or
                    receipt["sha256"] in control_digests):
                raise CodecError("Gaussian control alias")
            control_ids.add(value["control_id"])
            control_digests.add(receipt["sha256"])
            controls.append({**receipt, "control_id": value["control_id"]})

        read = _strict_json(payloads["routed_read_ledger"], "routed read ledger")
        amplification = read.get("maximum_routed_read_amplification")
        if (read.get("schema") != READ_SCHEMA or not isinstance(amplification,
                (int, float)) or isinstance(amplification, bool) or
                not (0 <= float(amplification) < 2.0)):
            raise CodecError("routed read ledger semantic receipt")

        audit = _strict_json(payloads["independent_audit_receipt"],
                             "independent audit receipt")
        if (audit.get("schema") != AUDIT_SCHEMA or audit.get("passed") is not True or
                audit.get("producer_source_manifest_sha256") !=
                self.expected_source_manifest_sha256 or
                audit.get("producer_source_root_sha256") !=
                self.expected_source_root_sha256):
            raise CodecError("independent audit semantic receipt")

        return {
            "authorized": True,
            "authenticated_objects": authenticated,
            "gaussian_controls": controls,
            "gaussian_control_count": len(controls),
            "maximum_routed_read_amplification": float(amplification),
            "producer_source_manifest_sha256":
                self.expected_source_manifest_sha256,
            "producer_source_root_sha256": self.expected_source_root_sha256,
            "digest_syntax_only_authority": False,
        }


def held_source_only_hooks() -> ProductionHooks:
    """The only hook object shipped by v2; payload launch must fail."""
    return ProductionHooks()
