#!/usr/bin/env python3
"""Fetch only the locked Qwen3-1.7B MLP tensors used by the upcycle screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path


MODEL = "Qwen/Qwen3-1.7B-Base"
REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors"
LAYERS = (9, 15)
ROLES = ("up_proj", "down_proj")
EXPECTED_TOTAL_BYTES = 3_441_185_608


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch_range(start: int, stop_inclusive: int) -> tuple[bytes, str | None]:
    request = urllib.request.Request(
        URL, headers={"Range": f"bytes={start}-{stop_inclusive}"}
    )
    with urllib.request.urlopen(request) as response:
        if response.status != 206:
            raise RuntimeError(f"range request returned HTTP {response.status}")
        content_range = response.headers.get("Content-Range")
        expected = stop_inclusive - start + 1
        value = response.read()
        if len(value) != expected:
            raise RuntimeError(f"short range: {len(value)} != {expected}")
        return value, content_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    prefix, content_range = fetch_range(0, 7)
    header_bytes = struct.unpack("<Q", prefix)[0]
    header_raw, _ = fetch_range(8, 8 + header_bytes - 1)
    header = json.loads(header_raw)
    data_start = 8 + header_bytes
    if content_range is None or not content_range.endswith(f"/{EXPECTED_TOTAL_BYTES}"):
        raise RuntimeError("unexpected checkpoint size")

    tensors = []
    for layer in LAYERS:
        for role in ROLES:
            name = f"model.layers.{layer}.mlp.{role}.weight"
            descriptor = header.get(name)
            if not isinstance(descriptor, dict):
                raise KeyError(name)
            if descriptor["dtype"] != "BF16":
                raise ValueError(f"unexpected dtype for {name}")
            expected_shape = [6144, 2048] if role == "up_proj" else [2048, 6144]
            if descriptor["shape"] != expected_shape:
                raise ValueError(f"unexpected shape for {name}")
            relative_start, relative_stop = map(int, descriptor["data_offsets"])
            raw, tensor_content_range = fetch_range(
                data_start + relative_start, data_start + relative_stop - 1
            )
            output_name = f"layer_{layer:02d}_{role}.bf16.bin"
            output_path = args.output_dir / output_name
            output_path.write_bytes(raw)
            tensors.append(
                {
                    "tensor": name,
                    "output": output_name,
                    "shape": expected_shape,
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                    "relative_data_offsets": [relative_start, relative_stop],
                    "absolute_http_range": [
                        data_start + relative_start,
                        data_start + relative_stop - 1,
                    ],
                    "content_range": tensor_content_range,
                }
            )
    manifest = {
        "schema": "qwen_dense_upcycle_locked_ranges_v1",
        "model": MODEL,
        "revision": REVISION,
        "url": URL,
        "checkpoint_total_bytes": EXPECTED_TOTAL_BYTES,
        "safetensors_header_bytes": header_bytes,
        "safetensors_header_sha256": sha256_bytes(header_raw),
        "layers": list(LAYERS),
        "roles": list(ROLES),
        "tensors": tensors,
    }
    unsigned = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    manifest["canonical_unsigned_sha256"] = sha256_bytes(unsigned)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
