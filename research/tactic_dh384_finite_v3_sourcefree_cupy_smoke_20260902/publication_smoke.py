#!/usr/bin/env python3
"""External fault/success checks for the frozen atomic publisher."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path


EXPECTED_MANIFEST = "bf0659d1fd6742768d14790ea980aa17321818d15e19ddd7d0dfaa8a223009b8"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    package = Path(sys.argv[1]).resolve(strict=True)
    manifest_payload = (package / "SOURCE_MANIFEST.json").read_bytes()
    assert sha256(manifest_payload) == EXPECTED_MANIFEST
    manifest = json.loads(manifest_payload)
    row = next(row for row in manifest["members"]
               if row["name"] == "atomic_publish.py")
    source = (package / row["name"]).read_bytes()
    assert len(source) == row["bytes"] and sha256(source) == row["sha256"]
    module = types.ModuleType("tactic_v3_external_publish_smoke")
    module.__file__ = f"<publish-smoke:{sha256(source)}>"
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True,
                 optimize=0), module.__dict__)
    with tempfile.TemporaryDirectory(prefix="tactic-v3-publish-") as raw:
        root = Path(raw).resolve(strict=True)
        final = root / "complete-result"
        receipt = module.publish_atomic(
            final, {"A.bin": b"alpha", "B.json": b"{}\n"},
            {"schema": "fixture", "status": "PASS"})
        assert final.is_dir()
        assert {path.name for path in final.iterdir()} == {
            "A.bin", "B.json", "COMPLETE.json"}
        complete = json.loads((final / "COMPLETE.json").read_bytes())
        assert complete["status"] == "PASS"
        before = {path.name: sha256(path.read_bytes()) for path in final.iterdir()}
        try:
            module.publish_atomic(final, {"X": b"x"},
                                  {"schema": "x", "status": "x"})
        except module.PublishError:
            pass
        else:
            raise AssertionError("existing output overwrite was accepted")
        after = {path.name: sha256(path.read_bytes()) for path in final.iterdir()}
        assert before == after

        fault = root / "fault-result"
        original = module._write_member
        calls = {"count": 0}

        def injected(directory_fd, name, payload):
            calls["count"] += 1
            if calls["count"] == 2:
                raise module.PublishError("injected prepublication fault")
            return original(directory_fd, name, payload)

        module._write_member = injected
        try:
            module.publish_atomic(
                fault, {"A": b"a", "B": b"b"},
                {"schema": "fault", "status": "fault"})
        except module.PublishError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("injected publication fault did not fire")
        finally:
            module._write_member = original
        assert not fault.exists()
        assert not any("fault-result.partial" in path.name
                       for path in root.iterdir())
        output = {
            "schema": "tactic-dh384-finite-v3-publication-smoke-v1",
            "status": "PASS_ATOMIC_SUCCESS_NO_OVERWRITE_AND_FAULT_CLEANUP",
            "source_manifest_sha256": EXPECTED_MANIFEST,
            "success_completion_sha256": before["COMPLETE.json"],
            "existing_target_unchanged": True,
            "prepublication_fault_left_no_public_or_staging_namespace": True,
            "publication_receipt": receipt,
            "qwen_or_model_payload_accessed": False,
            "v6_live_result_accessed": False,
        }
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
