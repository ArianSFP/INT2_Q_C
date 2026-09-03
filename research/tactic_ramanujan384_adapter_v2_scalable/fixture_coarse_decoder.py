#!/usr/bin/env python3
"""Source-free zero decoder used only by the sealed synthetic fixture."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class SourceFreeZeroCoarseDecoder:
    capability_id = "TACTIC_RAMANUJAN384_SOURCE_FREE_ZERO_COARSE_DECODER_V2"

    def __init__(self, expected_payload_sha256: str) -> None:
        self.expected_payload_sha256 = expected_payload_sha256

    def decode_literal(self, payload: bytes, intermediate: int, hidden: int,
                       role_order: Sequence[str]):
        if hashlib.sha256(payload).hexdigest() != self.expected_payload_sha256:
            raise ValueError("source-free coarse payload identity")
        return {
            role: np.zeros((intermediate, hidden), dtype="<f4") for role in role_order
        }
