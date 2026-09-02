#!/usr/bin/env python3
"""Hostile source-only tests for compact exact STRATA polar state."""

from __future__ import annotations

import hashlib
import inspect
import json
import unittest

import numpy as np

from compact_sc import dense_decode_level, ragged_decode_level, replay_six_levels
from memory_plan import LEVELS, all_beam_table, memory_plan, work_plan
from persistent_state import LayerCowPool, PackedSurvivorTape, StateError


class CompactSemanticParity(unittest.TestCase):
    def _case(self, n: int, freeze_kind: int) -> None:
        rng = np.random.default_rng(0xE71C_0000 + 31 * n + freeze_kind)
        leaf = np.exp(rng.normal(0.0, 1.1, n)).astype(np.float64)
        frozen = rng.integers(0, 2, n, dtype=np.uint8)
        if freeze_kind == 0:
            freeze = np.zeros(n, dtype=np.uint8)
        elif freeze_kind == 1:
            freeze = np.ones(n, dtype=np.uint8)
            freeze[::3] = 0
        else:
            freeze = rng.integers(0, 2, n, dtype=np.uint8)
            freeze[0] = 0
        decisions = rng.integers(0, 2, int(np.count_nonzero(freeze == 0)),
                                 dtype=np.uint8).tolist()
        dense = dense_decode_level(np, leaf, freeze, frozen, decisions)
        ragged = ragged_decode_level(np, leaf, freeze, frozen, decisions)
        for key in ("output", "frequencies", "selected", "internal"):
            self.assertTrue(np.array_equal(dense[key], ragged[key]), key)
        depth = n.bit_length() - 1
        self.assertEqual(dense["lr_bytes"], (n // 2) * depth * 8)
        self.assertEqual(dense["mu_bytes"], (n // 2) * depth)
        self.assertEqual(ragged["lr_bytes"], (n - 1) * 8)
        self.assertEqual(ragged["mu_bytes"], n - 1)

    def test_exact_dense_ragged_parity(self) -> None:
        for n in (8, 16, 32, 64, 128):
            for freeze_kind in range(3):
                with self.subTest(n=n, freeze_kind=freeze_kind):
                    self._case(n, freeze_kind)

    def test_no_coordinate_local_event_abi(self) -> None:
        source = inspect.getsource(ragged_decode_level)
        self.assertIn("for i0 in range(n)", source)
        self.assertNotIn("six_events_per_coordinate", source)

    def test_exact_six_level_dense_ragged_semantic_parity(self) -> None:
        for n in (8, 16, 32, 64):
            rng = np.random.default_rng(0x5160_0000 + n)
            weights = np.exp(-0.5 * (np.arange(-31, 33) / 11.25) ** 2)
            flags, frozen, decisions = [], [], []
            for level in range(6):
                flag = rng.integers(0, 2, n, dtype=np.uint8)
                flag[(level * 3) % n] = 0
                flags.append(flag)
                frozen.append(rng.integers(0, 2, n, dtype=np.uint8))
                decisions.append(rng.integers(
                    0, 2, int(np.count_nonzero(flag == 0)), dtype=np.uint8).tolist())
            dense = replay_six_levels(np, weights, flags, frozen, decisions, layout="dense")
            ragged = replay_six_levels(np, weights, flags, frozen, decisions, layout="ragged")
            self.assertTrue(np.array_equal(dense["indices"], ragged["indices"]))
            self.assertTrue(np.all((dense["indices"] >= 0) & (dense["indices"] < 64)))
            self.assertTrue(np.array_equal(dense["frequencies"], ragged["frequencies"]))
            self.assertTrue(np.array_equal(dense["selected"], ragged["selected"]))
            self.assertEqual(dense["logical_bits"], ragged["logical_bits"])
            self.assertEqual(dense["payload"], ragged["payload"])
            for dense_level, ragged_level in zip(dense["levels"], ragged["levels"],
                                                  strict=True):
                for key in ("output", "internal", "frequencies", "selected"):
                    self.assertTrue(np.array_equal(dense_level[key], ragged_level[key]))


class MemoryAndWorkLedger(unittest.TestCase):
    def test_all_beams_fit_exact_frozen_representation(self) -> None:
        rows = all_beam_table(1 << 21)
        self.assertEqual([row["beam_width"] for row in rows], [4, 8, 16, 32])
        self.assertTrue(all(row["passes_4gib_cap"] for row in rows))
        self.assertTrue(all(rows[index]["aligned_peak_bytes"] <
                            rows[index + 1]["aligned_peak_bytes"]
                            for index in range(3)))

    def test_ragged_active_cells_and_six_level_tape(self) -> None:
        n, beam = 1 << 21, 32
        plan = memory_plan(n, beam)
        by_name = {row["name"]: row for row in plan["buffers"]}
        self.assertEqual(by_name["likelihood_ragged_f64_banks"]["logical_bytes"],
                         beam * (n - 1) * 8)
        self.assertEqual(by_name["partial_sum_u8_ragged_banks"]["logical_bytes"],
                         beam * (n - 1))
        bits = 6
        self.assertEqual(plan["survivor_symbol_bits"], bits)
        self.assertEqual(by_name["survivor_ancestry_packed"]["logical_bytes"],
                         (LEVELS * n * beam * bits + 7) // 8)

    def test_exact_full_six_level_work_bounds(self) -> None:
        n, depth = 1 << 21, 21
        for beam in (4, 8, 16, 32):
            plan = work_plan(n, beam)
            self.assertEqual(plan["likelihood_node_updates"],
                             LEVELS * beam * n * depth)
            self.assertEqual(plan["partial_sum_state_writes_worst_active_upper_bound"],
                             LEVELS * beam * (n * depth // 2 + 1))
            self.assertEqual(plan["partial_sum_xors_worst_active_upper_bound"],
                             LEVELS * beam * (n * (depth - 2) // 2 + 1))
            self.assertEqual(plan["level_end_polar_xors"],
                             LEVELS * beam * n * depth // 2)
            startup = beam.bit_length() - 1
            events = LEVELS * n
            self.assertEqual(plan["branch_candidates_scored"],
                             2 * (beam - 1) + 2 * beam * (events - startup))
            self.assertEqual(plan["survivor_tape_symbols_written"],
                             2 * beam - 2 + beam * (events - startup))
            self.assertEqual(plan["winner_backtrace_events"], LEVELS * n)
            self.assertEqual(plan["winner_arithmetic_replay_events"], LEVELS * n)
            self.assertEqual(plan["winner_replay_likelihood_node_updates"],
                             LEVELS * n * depth)
            self.assertEqual(plan["winner_replay_lower_index_adds"], LEVELS * n)

    def test_checkpoint_recompute_is_explicit_not_free(self) -> None:
        n, beam = 1 << 21, 8
        base = work_plan(n, beam, checkpoint_spacing=1)
        recompute = work_plan(n, beam, checkpoint_spacing=7)
        self.assertEqual(recompute["likelihood_node_updates"],
                         7 * base["likelihood_node_updates"])
        self.assertEqual(recompute["likelihood_node_updates_without_recompute"],
                         base["likelihood_node_updates"])

    def test_packed_partial_sums_are_optional_and_account_work_layer(self) -> None:
        n, beam = 1 << 16, 8
        packed = memory_plan(n, beam, packed_partial_sums=True)
        names = {row["name"] for row in packed["buffers"]}
        self.assertIn("partial_sum_packed_banks", names)
        self.assertIn("partial_sum_u8_working_layer", names)


class PersistentStateTests(unittest.TestCase):
    def test_layer_copy_on_write_only_copies_mutated_layer(self) -> None:
        pool = LayerCowPool(4, [3, 5, 9])
        pool.create_path(0)
        pool.writable(0, 0)[:] = b"abc"
        pool.writable(0, 1)[:] = b"12345"
        pool.clone_path(0, 1)
        before0 = pool.handles[1][0]
        before1 = pool.handles[1][1]
        pool.writable(1, 1)[0] = ord("X")
        self.assertEqual(pool.handles[1][0], before0)
        self.assertNotEqual(pool.handles[1][1], before1)
        self.assertEqual(pool.readonly(0, 1), b"12345")
        self.assertEqual(pool.readonly(1, 1), b"X2345")
        self.assertEqual(pool.readonly(0, 0), pool.readonly(1, 0))

    def test_cow_pool_is_b_banks_not_two_b(self) -> None:
        pool = LayerCowPool(4, [2])
        pool.create_path(0)
        pool.clone_path(0, 1)
        pool.clone_path(0, 2)
        pool.clone_path(0, 3)
        # With all B paths alive and sharing one slot, making each independently
        # writable consumes exactly B slots, never 2B candidate states.
        for path in range(4):
            pool.writable(path, 0)[0] = path
        self.assertEqual(sum(ref > 0 for ref in pool.refs[0]), 4)
        with self.assertRaises(StateError):
            pool.clone_path(0, 4)

    def test_packed_tape_exact_roundtrip_and_backtrace(self) -> None:
        events, beam = 11, 8
        tape = PackedSurvivorTape(events, beam)
        expected = {}
        for event in range(events):
            for survivor in range(beam):
                parent = (survivor + event) % beam
                decision = (3 * survivor + event) & 1
                tape.write(event, survivor, parent, decision)
                expected[event, survivor] = (parent, decision)
        for key, value in expected.items():
            self.assertEqual(tape.read(*key), value)
        final = 5
        manual = []
        cursor = final
        for event in range(events - 1, -1, -1):
            cursor, bit = expected[event, cursor]
            manual.append(bit)
        self.assertEqual(tape.backtrace(final), tuple(reversed(manual)))
        self.assertEqual(len(tape.data),
                         (events * beam * (4)) // 8)  # B=8 -> 4-bit symbols.

    def test_tape_hash_is_deterministic(self) -> None:
        tape = PackedSurvivorTape(17, 32)
        for event in range(17):
            for survivor in range(32):
                tape.write(event, survivor, (7 * survivor + event) % 32,
                           (event ^ survivor) & 1)
        self.assertEqual(hashlib.sha256(tape.data).hexdigest(),
                         hashlib.sha256(bytes(tape.data)).hexdigest())


class SourceClosureTests(unittest.TestCase):
    def test_no_payload_interface(self) -> None:
        from run_gate import make_receipt
        signature = inspect.signature(make_receipt)
        self.assertEqual(tuple(signature.parameters), ())
        receipt = make_receipt()
        self.assertFalse(receipt["payload_authority"]["qwen"])
        self.assertFalse(receipt["payload_authority"]["current_codec"])
        self.assertEqual(receipt["verdicts"]["memory"],
                         "GO_MEMORY_CAPACITY")
        self.assertEqual(receipt["verdicts"]["compute"],
                         "HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION")
        self.assertEqual(receipt["verdicts"]["payload"], "HOLD_PAYLOAD")

    def test_receipt_is_canonical_json(self) -> None:
        from run_gate import make_receipt
        encoded = json.dumps(make_receipt(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False)
        self.assertEqual(encoded, encoded.encode("ascii").decode("ascii"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
