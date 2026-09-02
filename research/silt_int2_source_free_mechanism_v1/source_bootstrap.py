#!/usr/bin/env python3
"""Stdlib-only content-addressed bootstrap; imports nothing from the source tree."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
import tempfile


ROOT_FILES = (
    "POSTIMPLEMENTATION_REVIEW.md",
    "README.md",
    "cupy_backend_v1.py",
    "design_lock.json",
    "independent_decoder_v1.py",
    "run_synthetic_v1.py",
    "safe_publish.py",
    "silt_v1.py",
    "source_bootstrap.py",
    "test_source_only_v1.py",
    "verify_source_v1.py",
)


class BootstrapError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def open_absolute_directory_no_symlinks(directory: str) -> int:
    directory = os.path.abspath(directory)
    check(os.name == "posix" and os.path.isabs(directory), "absolute POSIX source directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in [value for value in directory.split("/") if value]:
            check(component not in (".", ".."), "canonical source path components")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def read_source_tree(directory: str) -> tuple[str, dict[str, bytes]]:
    descriptor = open_absolute_directory_no_symlinks(directory)
    try:
        observed_names = set(os.listdir(descriptor))
        check(observed_names == set(ROOT_FILES), "source tree must contain exactly the allowlisted files")
        packets: dict[str, bytes] = {}
        for name in sorted(ROOT_FILES):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            check(stat.S_ISREG(metadata.st_mode), f"source member {name} must be a regular file")
            file_descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_descriptor, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                packet = b"".join(chunks)
                check(len(packet) == metadata.st_size, f"source member {name} changed during read")
                packets[name] = packet
            finally:
                os.close(file_descriptor)
    finally:
        os.close(descriptor)
    hasher = hashlib.sha256()
    hasher.update(b"SILT-V1-SOURCE-ROOT\0")
    for name in sorted(ROOT_FILES):
        encoded = name.encode("utf-8")
        packet = packets[name]
        hasher.update(len(encoded).to_bytes(4, "big"))
        hasher.update(encoded)
        hasher.update(len(packet).to_bytes(8, "big"))
        hasher.update(hashlib.sha256(packet).digest())
    return hasher.hexdigest(), packets


def write_private_snapshot(directory: str, packets: dict[str, bytes]) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        for name in sorted(ROOT_FILES):
            file_descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            try:
                packet = packets[name]
                offset = 0
                while offset < len(packet):
                    written = os.write(file_descriptor, packet[offset:])
                    check(written > 0, "snapshot short write")
                    offset += written
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--expected-root")
    parser.add_argument("--print-observed-root", action="store_true")
    parser.add_argument("--entry", choices=("verify", "synthetic"), default="verify")
    parser.add_argument("entry_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    observed_root, packets = read_source_tree(arguments.source_dir)
    if arguments.print_observed_root:
        print(observed_root)
        return 0
    check(arguments.expected_root is not None and len(arguments.expected_root) == 64, "external expected root required")
    check(observed_root == arguments.expected_root.lower(), "source root mismatch before import")
    entry_name = "verify_source_v1.py" if arguments.entry == "verify" else "run_synthetic_v1.py"
    with tempfile.TemporaryDirectory(prefix="silt-v1-authenticated-snapshot-") as snapshot:
        os.chmod(snapshot, 0o700)
        write_private_snapshot(snapshot, packets)
        snapshot_root, _ = read_source_tree(snapshot)
        check(snapshot_root == observed_root, "authenticated snapshot root")
        forwarded_arguments = list(arguments.entry_arguments)
        if forwarded_arguments and forwarded_arguments[0] == "--":
            forwarded_arguments.pop(0)
        command = [
            sys.executable,
            "-I",
            "-B",
            os.path.join(snapshot, entry_name),
            "--authenticated-root",
            observed_root,
            *forwarded_arguments,
        ]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONNOUSERSITE": "1",
        }
        for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME"):
            if name in os.environ:
                environment[name] = os.environ[name]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
