# LOGIC-Q v1 capped adapter local exact-stage replay

This receipt records the fresh exact-member local replay of source root
`5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560`.

The first staged verifier call failed closed because it was mistakenly given
the invalid external pin `0`; `LOCAL_REPLAY.json` preserves that failure. The
manifest was then hashed from its literal staged bytes and verification,
33 hostile tests, and the source-free fixture all passed with `-I -B`. Neither
the staging tree nor the final source package contained `__pycache__` or `.pyc`
after replay.

This is not the required independent source audit and does not authorize Qwen
payload access.
