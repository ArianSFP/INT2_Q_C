# FOSP-ARX v4 native-prestart source contract

Status: **inert source-only repair; no source access or execution is
authorized**.

V4 is a distinct successor to the immutable FOSP-v3 source package. It does
not modify v3 or its independent audit. It preserves the exact v3 scientific
oracle bytes and protocol bytes while repairing the three pre-execution
problems anticipated by hostile review of v3's pre-execution boundary.

No model or Qwen path was traversed, no model/pinned/validation payload was
opened, and no NumPy, CuPy, CUDA, GPU, network, calibration, deployment, or
authorization operation was performed. This package contains no launcher
binary, interpreter image, runtime manifest, key, authorization builder, or
production entrypoint.

## Frozen v3 science

`scientific_oracle_v3.py` is byte-identical to v3
`free_order_oracle_v3.py` at SHA-256
`9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070`.
`scientific_protocol_v3.json` is byte-identical to v3 `protocol_lock.json` at
SHA-256
`f4660cb8876a749eb1635dbf010a8df6199e845b0517dd8b15039ac9cf1fd097`.

The scientific order is unchanged: full `3 x 3` Qwen pair panel with legal
FP16, gross relaxed necessary bound, eight identically processed controls,
corrected legal FP16 statistic, corrected-relaxed diagnostic only, then legal
survivor or optimization-gap ambiguity. There are nine coefficients on each
of 767 selected edges.

The 117,224 side bits, `0.024843004014756944 bpw` side charge,
`0.16096404744368115 bpw` required net saving,
`0.1858070514584381 bpw` required gross saving, `1.0x` logical read, and
`1.0054349308378698x` maximum cold-page read are unchanged.

The exact n=8 regression is also frozen:

```text
corrected relaxed s = 0
Q legal FP16 s      = 0.7995602818589078
control legal FP16  = 0.5885652320580218
corrected legal     = 0.21099504980088601
required gross s    = 0.1858070514584381
```

## Why the v3 Python bootstrap was insufficient

The v4 repair contract addresses three anticipated issues:

1. Replacement bootstrap or verifier bytes execute before their Python-level
   self-check and can counterfeit success.
2. The interpreter may load filesystem startup resources such as `encodings`
   before Python code authenticates the declared runtime tree.
3. A command named `python` is not an executable identity and may resolve
   through a mutable or symlinked path.

V4 therefore does not ask Python to establish the boundary that precedes
Python.

## Externally pinned native-launch contract

`native_launcher_contract.json` requires an external native executable with no
Python-runtime dependency. The source package does not contain that executable
and cannot mint its trusted digest. Before opening the contract, an external
trust root must establish that the launcher is the canonical, regular,
single-link, administrator-owned immutable file whose exact SHA-256 was pinned.

The required order is:

1. Verify the native launcher’s regular identity and external SHA-256 pin.
2. Authenticate the exact launcher-contract bytes.
3. Open no-follow, authenticate, and retain the exact bootstrap bytes before
   any bootstrap execution.
4. Authenticate the complete interpreter/runtime image and establish its
   administrator-owned platform immutability before creating a Python process.
5. Apply the environment, descriptor, and namespace closure.
6. Create the Python process from the already held interpreter identity.

This order closes bootstrap self-substitution and ensures that interpreter
startup resources are authenticated before they can execute. Authentication
after interpreter startup is terminally too late.

## Interpreter/runtime boundary

The runtime identity covers the interpreter and the complete recursive object
closure, including all built-in/frozen state and every filesystem resource
needed during startup. At minimum the exact interpreter, `codecs.py`,
`encodings/__init__.py`, and `io.py` identities must appear; these minimum rows
do not replace complete closure.

Every runtime member must be a regular, single-link, administrator-owned,
platform-immutable file with exact bytes. The unprivileged runtime must lack
write, unlink, relabel, replacement, and alias-creation capability. The
administrator and runtime principals are distinct. Administrator compromise is
outside this runtime threat boundary and requires an independent deployment
audit.

An ordinary mutable virtual environment does **not** satisfy this contract,
even when its current pathname hashes happen to match. A read-only-looking
path, version string, Python package lock file, or post-startup tree scan is not
a substitute for pre-startup immutable-image authentication.

## Correct invocation contract

A future independently built and audited deployment would invoke a regular
externally pinned native executable, for example:

```text
/srv/fosp-v4-sealed/bin/fosp4-native-launcher \
  --externally-pinned-launcher-sha256 <trusted-native-executable-digest> \
  --externally-pinned-contract-sha256 <trusted-contract-digest> \
  --contract /srv/fosp-v4-sealed/contracts/native_launcher_contract.json
```

The displayed pathname is illustrative, not trusted by itself. It must resolve
to the exact externally pinned regular immutable executable under the native
launcher’s platform contract. Do not replace it with a generic interpreter,
shell command, PATH lookup, symlink, or ordinary virtual-environment binary.
This source package cannot perform the invocation because it includes neither
the native executable nor an authenticated interpreter image.

## Source-only QA

`launch_contract.py` is an inert standard-library validation model. It can
serialize and validate synthetic declarations but cannot inspect production
privileges or start a process. `bootstrap_v4.py` is an authentication subject
and terminal placeholder; direct execution reports the missing native launcher
and exits 78.

`test_source_only.py` includes hostile bootstrap and launcher substitution,
external-pin mismatch, symlink launcher, generic-interpreter path, ordinary
mutable venv, unauthenticated pre-startup runtime, mutable image, substituted or
missing `encodings`, runtime link, and launch-order regressions. All fixtures
are inert bytes.

With a separately trusted regular Python test interpreter, the source-only
verifier entry is `verify_package.py`. The test interpreter is QA tooling only;
it is not the deployment invocation and cannot satisfy the native launcher
contract.

`PACKAGE_MANIFEST.json` covers every other member. Authenticate the externally
reported manifest SHA-256 before independent audit. A verifier PASS is evidence
only and never authorizes model access, calibration, deployment, GPU work, or
production.

## Authorization boundary

This package explicitly authorizes nothing. Source access, calibration,
deployment, model/Qwen access, GPU execution, production, and
self-authorization are all false. Any future work requires a new independent
PASS audit plus a separately built, externally pinned native launcher and an
independently authenticated immutable interpreter/runtime deployment. None is
provided here.
