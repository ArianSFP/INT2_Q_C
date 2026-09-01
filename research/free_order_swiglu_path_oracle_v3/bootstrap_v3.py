#!/usr/bin/env python3
"""Minimal pre-import bootstrap for the sealed FOSP-v3 source tree.

Security boundary: invoke with an independently trusted Python executable as
``python -I -S bootstrap_v3.py ...`` and externally pin both the interpreter
and package-manifest SHA-256 values.  This file intentionally imports only
``sys`` and the platform's built-in OS primitive module before closure.
"""

import sys as _sys


if not (_sys.flags.isolated and _sys.flags.no_site and _sys.flags.safe_path):
    raise SystemExit("FOSP_BOOTSTRAP_FAIL: require Python -I -S safe-path mode")

_os = __import__("nt" if _sys.platform == "win32" else "posix")
if getattr(getattr(_os, "__spec__", None), "origin", None) != "built-in":
    raise SystemExit("FOSP_BOOTSTRAP_FAIL: OS primitive module is not built-in")
if _sys.platform == "win32":
    _winapi = __import__("_winapi")
    _msvcrt = __import__("msvcrt")
    if (getattr(getattr(_winapi, "__spec__", None), "origin", None) != "built-in" or
            getattr(getattr(_msvcrt, "__spec__", None), "origin", None) != "built-in"):
        raise SystemExit("FOSP_BOOTSTRAP_FAIL: Windows handle primitive is not built-in")

# Close the ambient import path before examining any attacker-controlled tree.
_sys.path[:] = []

_MASK32 = 0xFFFFFFFF
_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1,
    0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786,
    0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
    0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
    0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A,
    0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)


def _ror(value, count):
    return ((value >> count) | (value << (32 - count))) & _MASK32


def _sha256(raw):
    """Small auditable SHA-256 with no module dependency."""
    data = bytearray(raw)
    bit_length = len(data) * 8
    data.append(0x80)
    while len(data) % 64 != 56:
        data.append(0)
    data.extend(bit_length.to_bytes(8, "big"))
    state = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    for offset in range(0, len(data), 64):
        schedule = [int.from_bytes(data[offset + 4 * i:offset + 4 * i + 4], "big")
                    for i in range(16)]
        for index in range(16, 64):
            x = schedule[index - 15]
            y = schedule[index - 2]
            s0 = _ror(x, 7) ^ _ror(x, 18) ^ (x >> 3)
            s1 = _ror(y, 17) ^ _ror(y, 19) ^ (y >> 10)
            schedule.append((schedule[index - 16] + s0 + schedule[index - 7] + s1) & _MASK32)
        a, b, c, d, e, f, g, h = state
        for index in range(64):
            upper = _ror(e, 6) ^ _ror(e, 11) ^ _ror(e, 25)
            choice = (e & f) ^ ((~e) & g)
            first = (h + upper + choice + _K[index] + schedule[index]) & _MASK32
            lower = _ror(a, 2) ^ _ror(a, 13) ^ _ror(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            second = (lower + majority) & _MASK32
            h, g, f, e, d, c, b, a = g, f, e, (d + first) & _MASK32, c, b, a, (first + second) & _MASK32
        state = [(left + right) & _MASK32 for left, right in zip(state, (a, b, c, d, e, f, g, h))]
    return "".join(format(value, "08x") for value in state)


def _fail(message):
    raise SystemExit("FOSP_BOOTSTRAP_FAIL: " + message)


def _absolute(path):
    if _sys.platform == "win32":
        return _os._getfullpathname(path)
    if path.startswith("/"):
        return path
    return _os.getcwd().rstrip("/") + "/" + path


def _dirname(path):
    normalized = path.replace("\\", "/")
    parent = normalized.rsplit("/", 1)[0]
    if _sys.platform == "win32":
        return parent.replace("/", "\\")
    return parent or "/"


def _join(parent, name):
    separator = "\\" if _sys.platform == "win32" else "/"
    return parent.rstrip("/\\") + separator + name


def _reparse(info):
    return bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _open_directory(path):
    if _sys.platform == "win32":
        # FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT.  Convert
        # the native handle to a Python descriptor solely for fstat/close.
        handle = _winapi.CreateFile(path, 0, 7, 0, 3, 0x02200000, 0)
        return _msvcrt.open_osfhandle(handle, _os.O_RDONLY)
    flags = _os.O_RDONLY | getattr(_os, "O_DIRECTORY", 0) | getattr(_os, "O_NOFOLLOW", 0)
    return _os.open(path, flags)


def _regular_bytes(path, expected_size=None):
    try:
        before_name = _os.lstat(path)
    except OSError as exc:
        _fail("cannot lstat " + path + ": " + str(exc))
    if before_name.st_mode & 0o170000 != 0o100000 or _reparse(before_name):
        _fail("nonregular object forbidden: " + path)
    flags = _os.O_RDONLY | getattr(_os, "O_BINARY", 0) | getattr(_os, "O_NOFOLLOW", 0)
    try:
        descriptor = _os.open(path, flags)
    except OSError as exc:
        _fail("cannot open regular member " + path + ": " + str(exc))
    try:
        before = _os.fstat(descriptor)
        if before.st_mode & 0o170000 != 0o100000 or _reparse(before):
            _fail("opened member is not regular: " + path)
        if expected_size is not None and before.st_size != expected_size:
            _fail("byte-count mismatch: " + path)
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = _os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                _fail("short read: " + path)
            chunks.append(chunk)
            remaining -= len(chunk)
        if _os.read(descriptor, 1):
            _fail("file grew while held: " + path)
        after = _os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            _fail("file identity changed while held: " + path)
        return b"".join(chunks)
    finally:
        _os.close(descriptor)


def _manifest_rows(raw):
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        _fail("package manifest is not ASCII")
    rows = {}
    for number, line in enumerate(lines, 1):
        pieces = line.split("  ")
        if len(pieces) != 3:
            _fail("malformed package manifest line " + str(number))
        digest, size_text, name = pieces
        if (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)
                or not size_text.isdigit() or not name or name in rows
                or "/" in name or "\\" in name or name in (".", "..")):
            _fail("malformed package manifest row " + str(number))
        rows[name] = (digest, int(size_text))
    return rows


def _snapshot_package(package, expected_manifest_sha256):
    manifest_name = "ARTIFACT_SHA256SUMS.txt"
    root_info = _os.lstat(package)
    if root_info.st_mode & 0o170000 != 0o040000 or _reparse(root_info):
        _fail("package root must be a real directory")
    directory_descriptor = _open_directory(package)
    held_root = _os.fstat(directory_descriptor)
    if (held_root.st_dev, held_root.st_ino) != (root_info.st_dev, root_info.st_ino):
        _os.close(directory_descriptor)
        _fail("package root changed while opening")
    try:
        names = _os.listdir(package)
    except OSError as exc:
        _os.close(directory_descriptor)
        _fail("cannot enumerate package: " + str(exc))
    if len(names) != len(set(names)):
        _fail("duplicate package directory entry")
    # Reject every directory, link/reparse point, socket, FIFO, device, or other
    # nonregular object before opening or executing any package member.
    for name in names:
        if not isinstance(name, str) or name in (".", "..") or "/" in name or "\\" in name:
            _fail("unsafe package member name")
        info = _os.lstat(_join(package, name))
        if info.st_mode & 0o170000 != 0o100000 or _reparse(info):
            _os.close(directory_descriptor)
            _fail("nonregular package member forbidden: " + name)
    manifest_raw = _regular_bytes(_join(package, manifest_name))
    if _sha256(manifest_raw) != expected_manifest_sha256:
        _fail("externally pinned package-manifest hash mismatch")
    rows = _manifest_rows(manifest_raw)
    if set(names) != set(rows) | {manifest_name}:
        _fail("package object closure mismatch")
    snapshot = {}
    for name in sorted(rows):
        digest, size = rows[name]
        raw = _regular_bytes(_join(package, name), size)
        if _sha256(raw) != digest:
            _os.close(directory_descriptor)
            _fail("package member hash mismatch: " + name)
        snapshot[name] = raw
    current_root = _os.lstat(package)
    if (_reparse(current_root) or
            (held_root.st_dev, held_root.st_ino) != (current_root.st_dev, current_root.st_ino)):
        _os.close(directory_descriptor)
        _fail("package root identity changed during snapshot")
    _os.close(directory_descriptor)
    return snapshot


def _runtime_rows(raw):
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        _fail("runtime manifest is not ASCII")
    if not lines or lines[0] != "FOSP_RUNTIME_CLOSURE_V1":
        _fail("wrong runtime manifest header")
    directories = set()
    files = {}
    import_roots = []
    for number, line in enumerate(lines[1:], 2):
        pieces = line.split("  ")
        if len(pieces) == 2 and pieces[0] in ("D", "I"):
            kind, relative = pieces
            if not relative or relative.startswith(("/", "\\")) or ".." in relative.replace("\\", "/").split("/"):
                _fail("unsafe runtime path line " + str(number))
            if kind == "D":
                if relative in directories:
                    _fail("duplicate runtime directory")
                directories.add(relative)
            else:
                import_roots.append(relative)
        elif len(pieces) == 4 and pieces[0] == "F":
            _, digest, size_text, relative = pieces
            if (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)
                    or not size_text.isdigit() or not relative or relative in files
                    or relative.startswith(("/", "\\"))
                    or ".." in relative.replace("\\", "/").split("/")):
                _fail("malformed runtime file line " + str(number))
            files[relative] = (digest, int(size_text))
        else:
            _fail("malformed runtime manifest line " + str(number))
    if not import_roots or len(import_roots) != len(set(import_roots)):
        _fail("runtime import-root closure invalid")
    if any(root not in directories for root in import_roots):
        _fail("runtime import root is not a declared directory")
    return directories, files, import_roots


def _scan_runtime(root):
    root_info = _os.lstat(root)
    if root_info.st_mode & 0o170000 != 0o040000 or _reparse(root_info):
        _fail("runtime root must be a real directory")
    directories = set()
    files = set()
    pending = [(root, ".")]
    while pending:
        parent, relative_parent = pending.pop()
        for name in _os.listdir(parent):
            relative = name if relative_parent == "." else relative_parent + "/" + name
            path = _join(parent, name)
            info = _os.lstat(path)
            kind = info.st_mode & 0o170000
            if _reparse(info):
                _fail("runtime link/special object forbidden: " + relative)
            if kind == 0o040000:
                directories.add(relative)
                pending.append((path, relative))
            elif kind == 0o100000:
                files.add(relative)
            else:
                _fail("runtime link/special object forbidden: " + relative)
    return directories, files


def _verify_runtime(root, manifest_path, expected_manifest_sha256, expected_python_sha256):
    if _sha256(_regular_bytes(_sys.executable)) != expected_python_sha256:
        _fail("externally pinned Python executable hash mismatch")
    manifest_raw = _regular_bytes(manifest_path)
    if _sha256(manifest_raw) != expected_manifest_sha256:
        _fail("externally pinned runtime-manifest hash mismatch")
    directories, files, import_roots = _runtime_rows(manifest_raw)
    observed_directories, observed_files = _scan_runtime(root)
    if observed_directories != directories or observed_files != set(files):
        _fail("runtime exact object closure mismatch")
    snapshot = {}
    for relative in sorted(files):
        digest, size = files[relative]
        raw = _regular_bytes(_join(root, relative.replace("/", "\\" if _sys.platform == "win32" else "/")), size)
        if _sha256(raw) != digest:
            _fail("runtime member hash mismatch: " + relative)
        snapshot[relative] = raw
    return import_roots, snapshot


class _SealedSourceFinder:
    """Import only authenticated Python source retained in memory."""

    def __init__(self, roots, snapshot):
        self.roots = tuple(root.rstrip("/\\") for root in roots)
        self.snapshot = snapshot
        self.selected = {}

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        relative_module = fullname.replace(".", "/")
        for root in self.roots:
            prefix = "" if root in ("", ".") else root + "/"
            package_name = prefix + relative_module + "/__init__.py"
            module_name = prefix + relative_module + ".py"
            if package_name in self.snapshot:
                self.selected[fullname] = (package_name, True)
                return _sys.modules["_frozen_importlib"].ModuleSpec(
                    fullname, self, is_package=True
                )
            if module_name in self.snapshot:
                self.selected[fullname] = (module_name, False)
                return _sys.modules["_frozen_importlib"].ModuleSpec(
                    fullname, self, is_package=False
                )
        return None

    def create_module(self, spec):
        del spec
        return None

    def exec_module(self, module):
        relative, is_package = self.selected.pop(module.__name__)
        label = "<sealed-runtime>/" + relative
        module.__file__ = label
        if is_package:
            module.__path__ = ["<sealed-runtime-package>/" + relative.rsplit("/", 1)[0]]
        source = self.snapshot[relative].decode("utf-8")
        exec(compile(source, label, "exec", dont_inherit=True), module.__dict__, module.__dict__)


def _parse_arguments(argv):
    options = {}
    forwarded = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            forwarded = argv[index + 1:]
            break
        if token == "--verify-package":
            options[token] = True
            index += 1
            continue
        if not token.startswith("--") or index + 1 >= len(argv):
            _fail("malformed bootstrap arguments")
        if token in options:
            _fail("duplicate bootstrap argument: " + token)
        options[token] = argv[index + 1]
        index += 2
    return options, forwarded


def _main():
    options, forwarded = _parse_arguments(_sys.argv[1:])
    expected_package = options.get("--package-manifest-sha256", "")
    if len(expected_package) != 64 or any(ch not in "0123456789abcdef" for ch in expected_package):
        _fail("package manifest SHA-256 is required")
    package = _dirname(_absolute(__file__))
    snapshot = _snapshot_package(package, expected_package)
    if options.get("--verify-package") is True:
        if len(options) != 2:
            _fail("verify-package mode accepts no runtime options")
        print("FOSP_V3_PACKAGE_SNAPSHOT_PASS files=" + str(len(snapshot)))
        return
    required = {
        "--package-manifest-sha256", "--runtime-root", "--runtime-manifest",
        "--runtime-manifest-sha256", "--python-sha256", "--entrypoint",
    }
    if set(options) != required:
        _fail("runtime mode argument closure mismatch")
    entrypoint = options["--entrypoint"]
    if entrypoint not in snapshot or not entrypoint.endswith(".py"):
        _fail("entrypoint is not a sealed package source")
    runtime_root = _absolute(options["--runtime-root"])
    runtime_manifest = _absolute(options["--runtime-manifest"])
    import_roots, runtime_snapshot = _verify_runtime(
        runtime_root,
        runtime_manifest,
        options["--runtime-manifest-sha256"],
        options["--python-sha256"],
    )
    # Filesystem import is removed entirely. Built-in/frozen modules remain;
    # every Python source module comes from authenticated in-memory bytes.
    sealed_finder = _SealedSourceFinder(import_roots, runtime_snapshot)
    frozen = _sys.modules["_frozen_importlib"]
    _sys.meta_path[:] = [frozen.BuiltinImporter, frozen.FrozenImporter, sealed_finder]
    _sys.path[:] = []
    _sys.argv[:] = [entrypoint] + forwarded
    namespace = {
        "__name__": "__main__",
        "__file__": "<sealed-package>/" + entrypoint,
        "__package__": None,
        "__cached__": None,
        "FOSP_BOOTSTRAP_PACKAGE_BYTES": snapshot,
        "FOSP_BOOTSTRAP_PACKAGE_MANIFEST_SHA256": expected_package,
    }
    source = snapshot[entrypoint].decode("utf-8")
    exec(compile(source, namespace["__file__"], "exec", dont_inherit=True), namespace, namespace)


_main()
