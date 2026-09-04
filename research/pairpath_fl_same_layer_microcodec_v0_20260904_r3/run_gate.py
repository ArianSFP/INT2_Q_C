"""Non-authority marker: r3 is source-only."""

PAYLOAD_EXECUTION_ENABLED = False
LOCAL_GPU_EXECUTION_ENABLED = False
QWEN_APERTURE_AUTHORIZED = False

if __name__ == "__main__":
    raise SystemExit("SOURCE_ONLY_HOLD_NO_PAYLOAD_OR_GPU_AUTHORITY")
