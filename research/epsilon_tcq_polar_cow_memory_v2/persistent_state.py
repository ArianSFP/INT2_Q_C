#!/usr/bin/env python3
"""Layer-granular COW handles and packed survivor ancestry for v2."""

from __future__ import annotations

import math
from typing import Sequence


class StateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StateError(message)


class LayerCowPool:
    """Tiny reference implementation of production layer-slot semantics."""

    def __init__(self, beam: int, layer_sizes: Sequence[int]) -> None:
        require(type(beam) is int and beam >= 2 and
                layer_sizes and all(type(value) is int and value > 0
                                    for value in layer_sizes), "COW geometry")
        self.beam = beam
        self.layer_sizes = tuple(layer_sizes)
        self.buffers = [[bytearray(size) for _ in range(beam)]
                        for size in self.layer_sizes]
        self.refs = [[0] * beam for _ in self.layer_sizes]
        self.handles: dict[int, list[int]] = {}

    def create_path(self, path: int) -> None:
        require(path not in self.handles and 0 <= path < self.beam,
                "new COW path")
        handles = []
        for layer in range(len(self.layer_sizes)):
            free = next((slot for slot, ref in enumerate(self.refs[layer]) if ref == 0), None)
            require(free is not None, "COW layer slot exhaustion")
            self.buffers[layer][free][:] = bytes(self.layer_sizes[layer])
            self.refs[layer][free] = 1
            handles.append(free)
        self.handles[path] = handles

    def clone_path(self, parent: int, child: int) -> None:
        require(parent in self.handles and child not in self.handles and
                0 <= child < self.beam, "COW clone path")
        handles = list(self.handles[parent])
        for layer, slot in enumerate(handles):
            self.refs[layer][slot] += 1
        self.handles[child] = handles

    def drop_path(self, path: int) -> None:
        require(path in self.handles, "COW drop path")
        for layer, slot in enumerate(self.handles.pop(path)):
            self.refs[layer][slot] -= 1
            require(self.refs[layer][slot] >= 0, "COW nonnegative reference")

    def writable(self, path: int, layer: int) -> bytearray:
        require(path in self.handles and 0 <= layer < len(self.layer_sizes),
                "COW writable request")
        slot = self.handles[path][layer]
        if self.refs[layer][slot] > 1:
            free = next((index for index, ref in enumerate(self.refs[layer]) if ref == 0), None)
            require(free is not None, "COW copy slot exhaustion")
            self.buffers[layer][free][:] = self.buffers[layer][slot]
            self.refs[layer][slot] -= 1
            self.refs[layer][free] = 1
            self.handles[path][layer] = free
            slot = free
        return self.buffers[layer][slot]

    def readonly(self, path: int, layer: int) -> bytes:
        require(path in self.handles and 0 <= layer < len(self.layer_sizes),
                "COW readonly request")
        return bytes(self.buffers[layer][self.handles[path][layer]])


class PackedSurvivorTape:
    def __init__(self, events: int, beam: int) -> None:
        require(type(events) is int and events > 0 and beam in (4, 8, 16, 32),
                "survivor tape geometry")
        self.events = events
        self.beam = beam
        self.parent_bits = int(math.log2(beam))
        self.symbol_bits = self.parent_bits + 1
        self.data = bytearray((events * beam * self.symbol_bits + 7) // 8)

    def _offset(self, event: int, survivor: int) -> int:
        require(0 <= event < self.events and 0 <= survivor < self.beam,
                "survivor tape coordinate")
        return (event * self.beam + survivor) * self.symbol_bits

    def write(self, event: int, survivor: int, parent: int, decision: int) -> None:
        require(0 <= parent < self.beam and decision in (0, 1),
                "survivor tape symbol")
        value = (parent << 1) | decision
        offset = self._offset(event, survivor)
        for bit in range(self.symbol_bits):
            absolute = offset + bit
            mask = 1 << (7 - (absolute & 7))
            if value & (1 << (self.symbol_bits - 1 - bit)):
                self.data[absolute >> 3] |= mask
            else:
                self.data[absolute >> 3] &= ~mask

    def read(self, event: int, survivor: int) -> tuple[int, int]:
        offset = self._offset(event, survivor)
        value = 0
        for bit in range(self.symbol_bits):
            absolute = offset + bit
            value = (value << 1) | ((self.data[absolute >> 3] >>
                                     (7 - (absolute & 7))) & 1)
        return value >> 1, value & 1

    def backtrace(self, final_survivor: int, event_count: int | None = None) -> tuple[int, ...]:
        count = self.events if event_count is None else int(event_count)
        require(0 <= final_survivor < self.beam and 0 <= count <= self.events,
                "survivor backtrace")
        cursor = final_survivor
        reversed_bits = []
        for event in range(count - 1, -1, -1):
            parent, decision = self.read(event, cursor)
            reversed_bits.append(decision)
            cursor = parent
        return tuple(reversed(reversed_bits))
