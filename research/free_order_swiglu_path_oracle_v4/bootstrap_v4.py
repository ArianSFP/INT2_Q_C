"""Inert bootstrap placeholder for the FOSP-v4 native-launch contract.

These bytes are an authentication subject, not a deployment entrypoint.  A
future externally pinned native launcher would have to authenticate them
before execution and authenticate the immutable interpreter/runtime before
Python startup.  This source package does neither and authorizes nothing.
"""

from __future__ import annotations


AUTHORITY_GRANTED = False
STATUS = "FOSP_V4_SOURCE_ONLY_NATIVE_LAUNCHER_ABSENT"


def source_only_status() -> dict[str, object]:
    return {
        "status": STATUS,
        "native_launcher_present": False,
        "runtime_authenticated_before_python_startup": False,
        "model_or_qwen_access": False,
        "gpu_access": False,
        "authorization": False,
    }


def _main() -> int:
    print(STATUS)
    return 78


if __name__ == "__main__":
    raise SystemExit(_main())
