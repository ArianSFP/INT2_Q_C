#!/usr/bin/env python3
"""Authenticated NumPy-scalar to STRATA ordinal bridge.

The producer and the independently frozen STRATA sources have deliberately
different scalar ABIs.  This tiny adapter is the only permitted crossing: an
integer NumPy scalar is converted to an exact built-in ``int`` and an ordered
ordinal vector is required to remain the canonical ``0..n-1`` sequence.  The
outer producer ABI and every row deliberately remain built-in ``list``
objects: NumPy treats a tuple of integers as multi-axis scalar indexing, while
a list of integers is one-axis advanced indexing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable


BRIDGE_ABI_BYTES = (
    b"UWFA-SC-V8-NUMPY-STRATA-ORDINAL-BRIDGE-V2\x00"
    b"numpy.integer-scalar=>builtin-int;exact-value;dtype-roundtrip;"
    b"source-order-preserved;unique-exact-coverage-0-through-n-minus-1;"
    b"outer-list-of-row-lists;row-list-preserves-one-axis-advanced-indexing;"
    b"tuple-row-forbidden"
)
BRIDGE_ABI_SHA256 = hashlib.sha256(BRIDGE_ABI_BYTES).hexdigest()


class OrdinalBridgeError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OrdinalBridgeError(message)


def numpy_scalar_to_builtin_int(np: Any, value: Any) -> int:
    """Convert one NumPy integer scalar without accepting coercible objects."""

    integer_type = getattr(np, "integer", None)
    bool_type = getattr(np, "bool_", ())
    _require(isinstance(integer_type, type), "NumPy integer scalar ABI unavailable")
    _require(isinstance(value, integer_type), "STRATA ordinal must be a NumPy integer scalar")
    _require(type(value) is not bool and not isinstance(value, bool_type), "boolean is not a STRATA ordinal")
    dtype = getattr(value, "dtype", None)
    _require(getattr(dtype, "kind", None) in {"i", "u"}, "STRATA ordinal dtype must be integral")
    item = value.item()
    _require(type(item) is int, "NumPy scalar .item() must produce a built-in int")
    converted = int(value)
    _require(type(converted) is int and converted == item, "NumPy scalar conversion changed value")
    dtype_type = getattr(dtype, "type", None)
    _require(callable(dtype_type), "NumPy scalar dtype round-trip unavailable")
    try:
        round_trip = dtype_type(item)
    except Exception as exc:
        raise OrdinalBridgeError(f"STRATA ordinal dtype round-trip failed: {exc}") from exc
    _require(isinstance(round_trip, integer_type) and int(round_trip) == item, "STRATA ordinal dtype round-trip changed value")
    return item


def bridge_canonical_ordinal_sequence(np: Any, values: Iterable[Any], *, count: int) -> list[int]:
    """Bridge to a built-in list while proving order and exact set coverage."""

    _require(type(count) is int and count >= 0, "STRATA ordinal count")
    _require(not isinstance(values, tuple), "tuple ordinal rows are forbidden by the one-axis indexing ABI")
    bridged = [numpy_scalar_to_builtin_int(np, value) for value in values]
    _require(len(bridged) == count, "STRATA ordinal cardinality changed")
    _require(len(set(bridged)) == len(bridged), "STRATA ordinal bridge produced duplicates")
    _require(set(bridged) == set(range(count)), "STRATA ordinal bridge changed exact coverage")
    return bridged


@dataclass(frozen=True)
class OrdinalBridgedStrataCommon:
    """Read-only proxy bridging the one NumPy-ordinal producing STRATA ABI."""

    wrapped: Any
    np: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def expected_block_group_ordinals(self, labels: Any) -> list[list[int]]:
        raw = self.wrapped.expected_block_group_ordinals(labels)
        _require(type(raw) is list, "STRATA ordinal outer container must be a built-in list")
        _require(all(type(block) is list for block in raw), "STRATA ordinal rows must be built-in lists")
        groups = int(self.wrapped.GROUPS)
        flattened = [value for block in raw for value in block]
        bridged = bridge_canonical_ordinal_sequence(self.np, flattened, count=groups)
        result: list[list[int]] = []
        cursor = 0
        for block in raw:
            width = len(block)
            result.append(list(bridged[cursor:cursor + width]))
            cursor += width
        _require(cursor == groups, "STRATA ordinal bridge partition coverage")
        return result


def wrap_strata_common(strata_common: Any, np: Any) -> OrdinalBridgedStrataCommon:
    _require(callable(getattr(strata_common, "expected_block_group_ordinals", None)), "STRATA ordinal producer ABI")
    _require(type(getattr(strata_common, "GROUPS", None)) is int, "STRATA GROUPS ABI must be built-in int")
    return OrdinalBridgedStrataCommon(strata_common, np)
