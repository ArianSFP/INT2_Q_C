"""No production aperture exists in the source-only r2 package."""

PAYLOAD_EXECUTION_ENABLED = False
LOCAL_GPU_EXECUTION_ENABLED = False
QWEN_APERTURE_AUTHORIZED = False


def main() -> None:
    raise SystemExit("SOURCE_ONLY_HOLD_NO_PAYLOAD_OR_GPU_AUTHORITY")


if __name__ == "__main__":
    main()
