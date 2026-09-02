#!/usr/bin/env python3
"""Fail-closed outer-codec hooks; deliberately unbound in this source gate."""

from __future__ import annotations

from dataclasses import dataclass
import re

from codec import CodecError


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProductionHooks:
    strata_packet_sha256: str | None = None
    scale_decoder_sha256: str | None = None
    forward_transform_sha256: str | None = None
    inverse_transform_sha256: str | None = None
    original_bf16_scorer_sha256: str | None = None
    gaussian_control_factory_sha256: str | None = None
    gaussian_control_count: int = 0
    component_framer_sha256: str | None = None
    routed_read_ledger_sha256: str | None = None
    independent_audit_receipt_sha256: str | None = None

    def authorize(self) -> dict:
        fields = {
            "strata_packet": self.strata_packet_sha256,
            "scale_decoder": self.scale_decoder_sha256,
            "forward_transform": self.forward_transform_sha256,
            "inverse_transform": self.inverse_transform_sha256,
            "original_bf16_scorer": self.original_bf16_scorer_sha256,
            "gaussian_control_factory": self.gaussian_control_factory_sha256,
            "component_framer": self.component_framer_sha256,
            "routed_read_ledger": self.routed_read_ledger_sha256,
            "independent_audit_receipt": self.independent_audit_receipt_sha256,
        }
        missing = [name for name, value in fields.items()
                   if not isinstance(value, str) or not SHA256_RE.fullmatch(value)]
        if missing:
            raise CodecError("unbound production hooks: " + ",".join(missing))
        if not isinstance(self.gaussian_control_count, int) or \
                self.gaussian_control_count < 8:
            raise CodecError("at least eight fully selected Gaussian controls")
        return {"authorized": True, "hooks": fields,
                "gaussian_control_count": self.gaussian_control_count}


def held_source_only_hooks() -> ProductionHooks:
    """The only hook object shipped by v1; payload launch must fail."""
    return ProductionHooks()
