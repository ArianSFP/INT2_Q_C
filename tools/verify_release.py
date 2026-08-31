#!/usr/bin/env python3
"""Offline, fail-closed verification of the compact publication artifacts.

This check needs only the Python standard library.  It binds the normative
codec files and primary evidence by SHA-256, reparses both physical Qwen
reservoirs with the independent unpacker, rederives every manifest seed, and
checks the published rate/distortion and Gaussian-confirmation invariants.
It does not recompute Qwen MSE, which requires the separately fetched BF16
source blocks and the full reproduction flow in docs/REPRODUCIBILITY.md.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BLOCKS = 32
BLOCK_LENGTH = 1 << 18
PHYSICAL_BYTES = 2_254_144
PHYSICAL_BITS = PHYSICAL_BYTES * 8
PHYSICAL_BPW = 2.14971923828125
GAUSSIAN_LIMIT = 0.050765774772264724
GAUSSIAN_CEILING = 0.053304063510877964
EXPECTED_DECODER_MAP_SHA256 = "a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef"

# Canonical-run files.  Support wrappers and human-authored documentation are
# intentionally excluded: they may evolve without changing the bitstream.
EXPECTED_SHA256 = {
    "src/polaris_sc_v2_encoder.py": "95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8",
    "src/polaris_sc_v2_rht_encoder.py": "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "src/reservoir_pack_v2.py": "c5fda34242153365dac07b5990bcd1fa19f0ac98d2512d47c3c8e1ec2a81dde8",
    "src/reservoir_unpack_v2.py": "cf7113c3fbc6340f0870dadcf7608739aa651f5706befa163b5d13516dac7e07",
    "src/independent_decoder_v1.py": "0652521fae5c77567e67cc9434adfb84b1a3cab53e5a79f250a3093a467be072",
    "src/agent_polaris_independent_decoder_v1.py": "0652521fae5c77567e67cc9434adfb84b1a3cab53e5a79f250a3093a467be072",
    "src/qwen_reservoir_decode.py": "2e1e484bf8ba98d493cfda55d4b23e275267e097e08907f5a9c606ae7350c797",
    "tools/run_qwen_panel.py": "4229ffd0d1fd43211ba2ad3022f1c684e452e4df158992493bd0f879663b3d59",
    "tools/audit_qwen_paired.py": "a29e4bd38dc5da88552d964af765e635cdd3571b4b3b4b160e361f36a2290943",
    "tools/audit_qwen_distribution.py": "643303fa4080590c409a55f6b1dc690d1fb68a36ce8c72b332345da77b6020ea",
    "results/qwen/manifest.json": "3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55",
    "results/qwen/exact_summary.json": "8b12e5bdbb8a58c4890637e7250697bcabdb83564037d9eb1e65a563bd7736a9",
    "results/qwen/rht_summary.json": "db4a70ab1d7dcf246d0232d580e329e3675b152f0f9c6232f3583434e43f3f5a",
    "results/qwen/paired_audit.json": "666fa77eb17fc9fda2c27f3081913b7a383d103acaf9b92749fcbbe921bcfedc",
    "results/qwen/distribution_audit.json": "f4724f0bd1118074b28ffed50d635a75640c5b926bf70777bfe7e1dc3e437fa0",
    "results/qwen/independent_audit.json": "52fc0af9cb95ccf6a0436f1a5710e3792f2b88e587070e27faec16149928029e",
    "results/qwen/independent_audit_erratum.json": "53660b5f3f3faee4c9c0d90fef70ac8c1d978800509029b72cd492b88a925738",
    "results/qwen/independent_audit_erratum.md": "3385f40faf01da94b24f6dcc488a1ecb8b58c038f5590ec8f5d1b4b1ae587da1",
    "results/qwen/exact_panel.plrsv2": "9388790c3cdbab5b9b33b676ced196090d81ba0422eb6fecfdd014bd2d054cf5",
    "results/qwen/rht_panel.plrsv2": "55d347c02ef1382ce209050d539f4e336dd7477125e4319e8b78d3067a436aac",
    "results/gaussian/result_card.json": "20c691dc932cef92619b6acf4a3072b66a7d69455c35fbb8a109e5663ec4152d",
    "results/gaussian/confirmation_summary.json": "f4988f8e92b99fa90a4fe2b6b153beb02a1cb123c49e21f64d90ce77f008e6b5",
    "results/gaussian/frozen_manifest.json": "06967a4e852c9d39c97fe39b45d50df558471e0f35912d330b6dc1e7493df5e0",
    "results/gaussian/confirmation.plrsv2": "ad0c35e72b5900ffa6ed353df1bf1b163d912b8bfb692fc8e5b318ea6f9eb3f5",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def same_float(observed: Any, expected: float, *, atol: float = 1e-15) -> bool:
    return math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=atol)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_bytes())
    require(isinstance(value, dict), f"{relative} must contain a JSON object")
    return value


def load_unpacker() -> Any:
    path = ROOT / "src/reservoir_unpack_v2.py"
    spec = importlib.util.spec_from_file_location("release_reservoir_unpacker", path)
    require(spec is not None and spec.loader is not None, "cannot load unpacker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_hashes() -> None:
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        require(path.is_file(), f"missing canonical artifact: {relative}")
        observed = sha256(path)
        require(observed == expected, f"SHA-256 mismatch for {relative}: {observed}")


def verify_optional_decoder_map() -> str:
    path = ROOT / "codec_data/polaris_sc_v1_decoder_map.npz"
    if not path.exists():
        return "not redistributed; generate locally for decoding"
    require(path.is_file(), "decoder map path exists but is not a regular file")
    observed = sha256(path)
    require(observed == EXPECTED_DECODER_MAP_SHA256, f"decoder map SHA-256 mismatch: {observed}")
    return "locally generated map present and hash-valid"


def verify_manifest() -> None:
    manifest = load_json("results/qwen/manifest.json")
    blocks = manifest.get("blocks")
    require(manifest.get("checkpoint") == CHECKPOINT, "checkpoint mismatch")
    require(manifest.get("revision") == REVISION, "revision mismatch")
    require(int(manifest.get("block_length", -1)) == BLOCK_LENGTH, "block length mismatch")
    require(isinstance(blocks, list) and len(blocks) == BLOCKS, "manifest block count mismatch")
    seen_ids: set[str] = set()
    seen_sc: set[int] = set()
    for row in blocks:
        identity = f"{REVISION}:{row['tensor']}:{int(row['canonical_block_index'])}"
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        expected_sc = int.from_bytes(digest[0:4], "big")
        expected_rht = int.from_bytes(digest[4:12], "big")
        require(int(row["sc_seed_u32"]) == expected_sc, f"SC seed mismatch: {row['id']}")
        require(int(row["rht_seed_u64"]) == expected_rht, f"RHT seed mismatch: {row['id']}")
        require(row["id"] not in seen_ids, f"duplicate block id: {row['id']}")
        require(expected_sc not in seen_sc, f"SC seed collision: {row['id']}")
        seen_ids.add(row["id"])
        seen_sc.add(expected_sc)


def verify_qwen_variant(
    unpacker: Any,
    variant: str,
    *,
    mse: float,
    logical_bits: int,
    headroom: int,
    reservoir_sha: str,
    expected_gate: bool,
) -> dict[str, Any]:
    summary = load_json(f"results/qwen/{variant}_summary.json")
    aggregate = summary["aggregate"]
    require(summary.get("status") == "complete", f"{variant}: incomplete status")
    require(summary.get("variant") == variant, f"{variant}: variant mismatch")
    require(summary.get("checkpoint") == CHECKPOINT, f"{variant}: checkpoint mismatch")
    require(summary.get("revision") == REVISION, f"{variant}: revision mismatch")
    require(int(summary.get("blocks", -1)) == BLOCKS, f"{variant}: block count mismatch")
    require(int(summary.get("block_length", -1)) == BLOCK_LENGTH, f"{variant}: block length mismatch")
    require(summary.get("manifest_sha256") == EXPECTED_SHA256["results/qwen/manifest.json"], f"{variant}: manifest hash mismatch")
    require(same_float(aggregate["energy_weighted_relative_mse"], mse), f"{variant}: MSE mismatch")
    require(int(aggregate["logical_bits_sum"]) == logical_bits, f"{variant}: logical bits mismatch")
    require(int(aggregate["payload_headroom_bits"]) == headroom, f"{variant}: headroom mismatch")
    require(int(aggregate["emitted_sample_reservoir_bytes"]) == PHYSICAL_BYTES, f"{variant}: physical bytes mismatch")
    require(same_float(aggregate["emitted_sample_reservoir_bpw_including_96_byte_header"], PHYSICAL_BPW), f"{variant}: bpw mismatch")
    require(bool(aggregate["physical_rate_at_most_2p15"]), f"{variant}: physical rate gate failed")
    require(bool(aggregate["payload_fits_global_reservoir"]), f"{variant}: payload overflow")
    require(bool(aggregate["all_independent_decodes_passed"]), f"{variant}: decode aggregate failed")
    require(bool(aggregate["passes_joint_rate_and_mse_gate"]) is expected_gate, f"{variant}: joint gate mismatch")
    require(bool(aggregate["passes_5pct_gaussian_mse_gate"]) is expected_gate, f"{variant}: MSE gate mismatch")
    details = summary.get("blocks_detail")
    require(isinstance(details, list) and len(details) == BLOCKS, f"{variant}: detail count mismatch")
    require(all(row.get("decoder_passed") is True for row in details), f"{variant}: a decoder failed")
    require(PHYSICAL_BITS * 20 <= 43 * (BLOCKS * BLOCK_LENGTH), f"{variant}: integer rate gate failed")

    path = ROOT / f"results/qwen/{variant}_panel.plrsv2"
    parsed = unpacker.validate_reservoir(path)
    require(parsed.file_bytes == PHYSICAL_BYTES, f"{variant}: parsed size mismatch")
    require(parsed.block_count == BLOCKS, f"{variant}: parsed block count mismatch")
    require(parsed.payload_logical_bits == logical_bits, f"{variant}: parsed logical bits mismatch")
    require(parsed.payload_unused_zero_bits == headroom, f"{variant}: parsed zero suffix mismatch")
    require(parsed.reservoir_sha256 == reservoir_sha, f"{variant}: parsed reservoir hash mismatch")
    require(summary["hashes"]["reservoir"] == reservoir_sha, f"{variant}: summary reservoir hash mismatch")
    return summary


def verify_qwen() -> None:
    unpacker = load_unpacker()
    exact = verify_qwen_variant(
        unpacker,
        "exact",
        mse=0.06319873774126093,
        logical_bits=17_916_908,
        headroom=113_940,
        reservoir_sha=EXPECTED_SHA256["results/qwen/exact_panel.plrsv2"],
        expected_gate=False,
    )
    rht = verify_qwen_variant(
        unpacker,
        "rht",
        mse=0.05289448474927123,
        logical_bits=18_006_314,
        headroom=24_534,
        reservoir_sha=EXPECTED_SHA256["results/qwen/rht_panel.plrsv2"],
        expected_gate=True,
    )
    exact_mse = float(exact["aggregate"]["energy_weighted_relative_mse"])
    rht_mse = float(rht["aggregate"]["energy_weighted_relative_mse"])
    require(same_float((exact_mse - rht_mse) / exact_mse, 0.163045234133882, atol=2e-15), "paired improvement mismatch")
    require(rht_mse <= GAUSSIAN_CEILING, "RHT is outside the five-percent ceiling")
    require(exact_mse > GAUSSIAN_CEILING, "exact control unexpectedly passes")
    require(same_float(float(rht["aggregate"]["gaussian_limit_mse_at_2p15"]), GAUSSIAN_LIMIT), "Gaussian limit mismatch")


def verify_gaussian() -> None:
    card = load_json("results/gaussian/result_card.json")
    require(card.get("status") == "passed frozen matched-Gaussian confirmation", "Gaussian status mismatch")
    require(card.get("strict_ptq") is True and card.get("training_or_retraining") is False, "Gaussian PTQ claim mismatch")
    envelope = card["whole_checkpoint_budget_envelope"]
    require(int(envelope["parameters"]) == 30_532_122_624, "envelope parameter count mismatch")
    require(int(envelope["total_bits"]) == 65_641_547_920, "envelope bit count mismatch")
    require(float(envelope["bpw"]) <= 2.15, "envelope exceeds 2.15 bpw")
    confirmation = card["confirmation"]
    require(int(confirmation["blocks"]) == BLOCKS, "Gaussian block count mismatch")
    require(int(confirmation["block_length"]) == BLOCK_LENGTH, "Gaussian block length mismatch")
    require(all(confirmation[key]["passed"] is True for key in ("logical_rate", "absolute_mse", "sample_relative_mse")), "a Gaussian confidence gate failed")
    require(float(confirmation["sample_relative_mse"]["ucb"]) <= GAUSSIAN_CEILING, "Gaussian UCB exceeds target")
    reservoir = card["serialized_confirmation_reservoir"]
    require(int(reservoir["physical_bytes"]) == PHYSICAL_BYTES, "Gaussian reservoir size mismatch")
    require(int(reservoir["zero_reserve_bits"]) == 23_270, "Gaussian zero reserve mismatch")
    require(reservoir["sha256"] == EXPECTED_SHA256["results/gaussian/confirmation.plrsv2"], "Gaussian reservoir hash mismatch")


def main() -> None:
    verify_hashes()
    decoder_map_status = verify_optional_decoder_map()
    verify_manifest()
    verify_qwen()
    verify_gaussian()
    print(
        json.dumps(
            {
                "status": "release artifacts verified",
                "hashes_verified": len(EXPECTED_SHA256),
                "decoder_map": decoder_map_status,
                "qwen": {
                    "blocks": BLOCKS,
                    "weights": BLOCKS * BLOCK_LENGTH,
                    "exact_gate": "FAIL (expected control)",
                    "rht_gate": "PASS",
                    "rht_mse": 0.05289448474927123,
                    "physical_bpw": PHYSICAL_BPW,
                },
                "gaussian_confirmation": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
