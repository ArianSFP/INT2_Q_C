#!/usr/bin/env python3
"""Frozen legal-label adapter ABI for epsilon-TCQ v0.

This module is standard-library only. The synthetic STRATA-like adapter is a
test fixture, never a Qwen/POLARIS integration.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


STRATA_INTERFACE = "strata_sc_6bit_legal_replay"
DIRECT4_INTERFACE = "direct_int2_4level_new_codec"
INTERFACES = (STRATA_INTERFACE, DIRECT4_INTERFACE)
CONTEXTS = 16


class InterfaceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InterfaceError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LegalChoice:
    label: int
    nominal: float
    event_bits: tuple[int, ...]
    event_contexts: tuple[int, ...]
    next_legal_state: int

    def validate(self, interface: str) -> None:
        require(interface in INTERFACES, "known interface")
        labels = 64 if interface == STRATA_INTERFACE else 4
        events = 6 if interface == STRATA_INTERFACE else 2
        require(type(self.label) is int and 0 <= self.label < labels,
                "legal label range")
        require(math.isfinite(self.nominal), "finite nominal reconstruction")
        require(len(self.event_bits) == len(self.event_contexts) == events,
                "literal event arity")
        require(all(bit in (0, 1) for bit in self.event_bits),
                "binary events")
        require(all(type(context) is int and 0 <= context < CONTEXTS
                    for context in self.event_contexts), "public contexts")
        require(type(self.next_legal_state) is int and
                0 <= self.next_legal_state < (1 << 31), "legal state bound")


class LegalAdapter(Protocol):
    interface: str
    decoder_replayable: bool
    source_identity_inputs: bool

    def initial_state(self) -> int: ...

    def encode_choices(
        self, position: int, legal_state: int, nearest_label: int, epsilon: int,
    ) -> Sequence[LegalChoice]: ...

    def decode_events(
        self, position: int, legal_state: int, event_bits: Sequence[int],
    ) -> LegalChoice: ...

    def decode_context(
        self, position: int, legal_state: int,
        partial_event_bits: Sequence[int],
    ) -> int: ...


def validate_adapter(adapter: LegalAdapter) -> dict[str, Any]:
    require(getattr(adapter, "interface", None) in INTERFACES,
            "adapter interface")
    require(getattr(adapter, "decoder_replayable", None) is True,
            "adapter decoder replayability")
    require(getattr(adapter, "source_identity_inputs", None) is False,
            "adapter source identity forbidden")
    initial = adapter.initial_state()
    require(type(initial) is int and 0 <= initial < (1 << 31),
            "adapter initial state")
    return {
        "interface": adapter.interface,
        "labels": 64 if adapter.interface == STRATA_INTERFACE else 4,
        "events_per_label": 6 if adapter.interface == STRATA_INTERFACE else 2,
        "initial_state": initial,
        "decoder_replayable": True,
        "source_identity_inputs": False,
        "direct4_is_not_strata_alias": adapter.interface != DIRECT4_INTERFACE
            or type(adapter).__name__ == "DirectFourLevelAdapter",
    }


def label_bits(label: int, width: int) -> tuple[int, ...]:
    require(type(label) is int and type(width) is int and width > 0 and
            0 <= label < 1 << width, "label bit conversion")
    return tuple((label >> shift) & 1 for shift in range(width - 1, -1, -1))


def bits_label(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        require(bit in (0, 1), "binary label bits")
        value = (value << 1) | int(bit)
    return value


class DirectFourLevelAdapter:
    """A literal new four-level codec, never a STRATA compatibility shim."""

    interface = DIRECT4_INTERFACE
    decoder_replayable = True
    source_identity_inputs = False

    def __init__(self, reproduction: Sequence[float]) -> None:
        require(len(reproduction) == 4 and
                all(math.isfinite(float(value)) for value in reproduction),
                "four finite reproduction levels")
        require(all(float(reproduction[index]) < float(reproduction[index + 1])
                    for index in range(3)), "strict reproduction order")
        self.reproduction = tuple(float(value) for value in reproduction)

    def initial_state(self) -> int:
        return 0

    def _choice(self, position: int, label: int) -> LegalChoice:
        bits = label_bits(label, 2)
        contexts = tuple((2 * (position & 3) + level) % CONTEXTS
                         for level in range(2))
        return LegalChoice(label, self.reproduction[label], bits, contexts, 0)

    def encode_choices(
        self, position: int, legal_state: int, nearest_label: int, epsilon: int,
    ) -> tuple[LegalChoice, ...]:
        require(legal_state == 0 and 0 <= nearest_label < 4 and epsilon in (1, 2),
                "direct4 encode request")
        lower = max(0, nearest_label - epsilon)
        upper = min(3, nearest_label + epsilon)
        return tuple(self._choice(position, label)
                     for label in range(lower, upper + 1))

    def decode_events(
        self, position: int, legal_state: int, event_bits: Sequence[int],
    ) -> LegalChoice:
        require(legal_state == 0 and len(event_bits) == 2,
                "direct4 decode request")
        return self._choice(position, bits_label(event_bits))

    def decode_context(
        self, position: int, legal_state: int,
        partial_event_bits: Sequence[int],
    ) -> int:
        require(legal_state == 0 and len(partial_event_bits) < 2,
                "direct4 context request")
        return (2 * (position & 3) + len(partial_event_bits)) % CONTEXTS


class SyntheticStrataLegalAdapter:
    """Source-free six-event legality fixture; not current-codec evidence."""

    interface = STRATA_INTERFACE
    decoder_replayable = True
    source_identity_inputs = False

    def __init__(self, scale: float = 0.25) -> None:
        require(math.isfinite(scale) and scale > 0.0, "synthetic scale")
        self.scale = float(scale)

    def initial_state(self) -> int:
        return 0

    def _legal(self, position: int, legal_state: int, label: int) -> bool:
        # The fixture exercises a stateful legal decoder without pretending to
        # reproduce the actual POLARIS constraint graph. Every in-range label
        # is legal; the authenticated production adapter must replace this.
        return 0 <= label < 64

    def _choice(self, position: int, legal_state: int, label: int) -> LegalChoice:
        bits = label_bits(label, 6)
        contexts = tuple((2 * level + (position & 1)) % CONTEXTS
                         for level in range(6))
        next_state = (legal_state * 17 + label * 3 + position + 1) & 255
        return LegalChoice(
            label, self.scale * (label - 31), bits, contexts, next_state)

    def encode_choices(
        self, position: int, legal_state: int, nearest_label: int, epsilon: int,
    ) -> tuple[LegalChoice, ...]:
        require(0 <= nearest_label < 64 and epsilon in (1, 2),
                "synthetic STRATA request")
        candidates = []
        for label in range(max(0, nearest_label - epsilon),
                           min(63, nearest_label + epsilon) + 1):
            if self._legal(position, legal_state, label) or label == nearest_label:
                choice = self._choice(position, legal_state, label)
                choice.validate(self.interface)
                candidates.append(choice)
        require(any(row.label == nearest_label for row in candidates),
                "nearest legal fallback")
        return tuple(candidates)

    def decode_events(
        self, position: int, legal_state: int, event_bits: Sequence[int],
    ) -> LegalChoice:
        require(len(event_bits) == 6, "synthetic STRATA event arity")
        label = bits_label(event_bits)
        require(self._legal(position, legal_state, label),
                "synthetic STRATA decoded path legality")
        return self._choice(position, legal_state, label)

    def decode_context(
        self, position: int, legal_state: int,
        partial_event_bits: Sequence[int],
    ) -> int:
        require(len(partial_event_bits) < 6,
                "synthetic STRATA context request")
        return (2 * len(partial_event_bits) + (position & 1)) % CONTEXTS


def assert_no_interface_alias(
    strata_choice: LegalChoice, direct_choice: LegalChoice,
) -> None:
    strata_choice.validate(STRATA_INTERFACE)
    direct_choice.validate(DIRECT4_INTERFACE)
    require(len(strata_choice.event_bits) == 6 and
            len(direct_choice.event_bits) == 2,
            "interfaces retain literal event widths")
    require(not (strata_choice.label == direct_choice.label and
                 strata_choice.event_bits == direct_choice.event_bits),
            "direct4/STRATA alias forbidden")
