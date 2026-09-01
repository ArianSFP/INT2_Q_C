from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common
import overlay


def _old_representatives(count: int) -> np.ndarray:
    values = []
    seed = 0
    while len(values) < count:
        for pp in (0, 2, 3):
            for ep in range(8):
                for etp in range(3):
                    for assignment in ((0,) if ep in (0, 7) else (0, 1)):
                        for half in range(2):
                            values.append(common.v4_logical_ordinal(
                                seed, pp, ep, etp, assignment, half, 0
                            ))
        seed += 1
    return np.asarray(values[:count], dtype=np.uint64)


class OverlayTests(unittest.TestCase):
    def test_v4_translation_is_strict_and_new_family_is_disjoint(self):
        old = _old_representatives(8 * 252)
        translated = np.asarray(
            [common.translate_v4_ordinal(int(value)) for value in old], dtype=np.uint64
        )
        new = common.representative_ordinals(0, 8)
        self.assertTrue(np.all(translated[1:] > translated[:-1]))
        self.assertEqual(np.intersect1d(translated, new).size, 0)
        self.assertEqual(np.union1d(translated, new).size, 8 * 896)
        with self.assertRaises(common.ProtocolError):
            common.translate_v4_ordinal(common.v4_logical_ordinal(0, 1, 0, 0, 0, 0, 0))
        with self.assertRaises(common.ProtocolError):
            common.translate_v4_ordinal(common.v4_logical_ordinal(0, 0, 0, 0, 1, 0, 0))

    def test_exact_merge_retains_all_4096_and_uses_metric_then_ordinal(self):
        old_o = np.tile(np.arange(0, 4096, 2, dtype=np.uint64), (33, 1))
        new_o = np.tile(np.arange(1, 4096, 2, dtype=np.uint64), (33, 1))
        old_q = np.zeros((33, 2048), dtype=np.float64)
        new_q = np.zeros((33, 2048), dtype=np.float64)
        merged = overlay.merge_topk(old_o, old_q, new_o, new_q)
        self.assertEqual(merged.domain_ordinals.shape, (33, 4096))
        self.assertTrue(np.array_equal(merged.domain_ordinals[0], np.arange(4096)))
        self.assertEqual(len(merged.union_ordinals), 4096)
        self.assertTrue(merged.receipt["no_post_merge_topk_truncation"])

    def test_merge_rejects_overlap_duplicate_nonfinite_and_wrong_dtype(self):
        old_o = np.tile(np.arange(2048, dtype=np.uint64), (33, 1))
        new_o = np.tile(np.arange(2048, 4096, dtype=np.uint64), (33, 1))
        q = np.tile(np.arange(2048, dtype=np.float64), (33, 1))
        bad = new_o.copy(); bad[0, 0] = old_o[0, 0]
        with self.assertRaises(common.ProtocolError):
            overlay.merge_topk(old_o, q, bad, q)
        bad_q = q.copy(); bad_q[0, 0] = np.nan
        with self.assertRaises(common.ProtocolError):
            overlay.merge_topk(old_o, bad_q, new_o, q)
        with self.assertRaises(common.ProtocolError):
            overlay.merge_topk(old_o.astype(np.int64), q, new_o, q)

    def _fixture(self, root: Path):
        lock = json.loads((common.PACKAGE_DIR / "candidate_lock.json").read_text())
        lock = copy.deepcopy(lock)
        binding = lock["v4_reuse"]
        binding["result_audit"]["required_fields"] = {}

        run = root / "run"
        (run / "state" / "events").mkdir(parents=True)
        (run / "state" / "files").mkdir()
        audit_path = root / "audit.json"

        base = _old_representatives(2048)
        ordinals = np.tile(base, (33, 1))
        metrics = np.tile(np.arange(2048, dtype=np.float64), (33, 1))
        state_path = run / "state" / "files" / "stage0_merged_global.npz"
        with state_path.open("wb") as handle:
            np.savez_compressed(
                handle, domain_top_ordinals=ordinals, domain_top_q=metrics,
                union_ordinals=np.unique(ordinals.reshape(-1)),
            )
        binding["merged_state_arrays"]["union_ordinals"]["shape"] = [2048]
        event = {
            "sequence": 257, "previous_event_sha256": "e" * 64,
            "kind": "stage0_merged", "key": "global",
            "relative_path": "files/stage0_merged_global.npz",
            "file_sha256": common.sha256_file(state_path),
            "file_bytes": state_path.stat().st_size, "created_unix_ns": 1,
        }
        event_path = run / "state" / "events" / "000257.json"
        event_path.write_text(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        binding["merged_event"] = {
            "relative_path": "state/events/000257.json",
            "bytes": event_path.stat().st_size,
            "sha256": common.sha256_file(event_path),
            "required_fields": event,
        }

        result = {
            "schema": "qwen3_initialization_anchor_tier_c_grouped_v4_result_v1",
            "bindings": binding["result_required_bindings"],
            "candidate_space": {"logical_candidate_count": 50_331_648,
                                "effective_candidate_count": 16_515_072,
                                "domain_ids": list(common.DOMAIN_IDS),
                                "equivalence_map_sha256": "1699afb9596faf197971c704f16aefd2a20e39e267f2f81ea41cc94a1a46e1e5"},
            "coordinates": {"stage0_plan_sha256": binding["stage0_plan_sha256"],
                            "full_plan_sha256": binding["full_plan_sha256"]},
            "search": {"stage0_top_k_per_domain": 2048, "stage0_shard_count": 256,
                       "union_shortlist_count": 2048},
            "physical_ledger": {"scientific_scores_use_decoded_fp16_affines": True},
            "backend": {"source_free_calibration": {
                "schema": "qwen3_initialization_anchor_tier_c_grouped_source_free_calibration_v4",
                "receipt_sha256": binding["calibration_internal_sha256"],
                "coordinate_count": 512, "candidate_count": 64_512,
            }},
            "resume_state": {"event_count_before_result": 392,
                             "events": [{} for _ in range(392)]},
        }
        result["resume_state"]["events"][257] = event
        result_path = run / binding["result_basename"]
        result_path.write_text(json.dumps(result, sort_keys=True))
        binding["result_bytes"] = result_path.stat().st_size
        binding["result_sha256"] = common.sha256_file(result_path)
        audit = {"schema": binding["result_audit"]["schema"],
                 "status": binding["result_audit"]["status"]}
        internal = common.sha256_bytes(common.canonical_json_bytes(audit))
        audit["audit_receipt_sha256"] = internal
        binding["result_audit"]["internal_sha256"] = internal
        binding["result_audit"]["required_fields"] = {"audit_receipt_sha256": internal}
        audit_path.write_text(json.dumps(audit, sort_keys=True))
        binding["result_audit"]["file_bytes"] = audit_path.stat().st_size
        binding["result_audit"]["file_sha256"] = common.sha256_file(audit_path)
        translated = np.asarray(
            [common.translate_v4_ordinal(int(value)) for value in ordinals.reshape(-1)],
            dtype=np.uint64,
        ).reshape(ordinals.shape)
        array_hashes = {
            "old_domain_top_ordinals_sha256_u64le": common.sha256_bytes(
                np.asarray(ordinals, dtype="<u8").tobytes()
            ),
            "translated_domain_top_ordinals_sha256_u64le": common.sha256_bytes(
                np.asarray(translated, dtype="<u8").tobytes()
            ),
            "domain_top_metrics_sha256_f64le": common.sha256_bytes(
                np.asarray(metrics, dtype="<f8").tobytes()
            ),
        }
        binding["authenticated_array_hashes"] = array_hashes
        receipt = {
            "schema": "qwen3_tier_c_grouped_v5_authenticated_v4_topk_v1",
            "result_sha256": binding["result_sha256"],
            "result_audit_sha256": binding["result_audit"]["file_sha256"],
            "merged_event_sha256": binding["merged_event"]["sha256"],
            "merged_state_sha256": event["file_sha256"],
            **array_hashes,
            "old_union_count": 2048,
            "translation_strictly_increasing_on_old_union": True,
            "old_topk_total_order_preserved": True,
            "qwen_payload_opened_by_authentication": False,
        }
        receipt["receipt_sha256"] = common.sha256_bytes(common.canonical_json_bytes(receipt))
        binding["expected_authentication_receipt_sha256"] = receipt["receipt_sha256"]
        return lock, run, audit_path, state_path

    def test_complete_authenticated_fixture_and_tamper_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, run, audit_path, state_path = self._fixture(Path(directory))
            authenticated = overlay.authenticate_v4_topk(run, audit_path, lock)
            self.assertEqual(authenticated.translated_ordinals.shape, (33, 2048))
            self.assertTrue(authenticated.receipt["translation_strictly_increasing_on_old_union"])
            raw = state_path.read_bytes()
            state_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(common.ProtocolError):
                overlay.authenticate_v4_topk(run, audit_path, lock)


if __name__ == "__main__":
    unittest.main()
