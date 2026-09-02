# UWFA-SC v9 universal BF16 source-moment contract v0

## Outcome

This source-only package closes the missing external-input ABI for
`uwfa_sc_v9_matched_gaussian_ptq_producer_v0`. It does not contain source
moments or payload bytes, and it does not authorize a source read. It defines
and tests the gate which a later reviewed dispatcher must use to publish the
producer's exact 18-row `uwfa-sc-v9-bf16-matrix-moment-contract-v1` input.

The runtime record is universal. Its only matrix identity is:

```text
(matrix ordinal, canonical slot 0..5, SwiGLU role, storage shape)
```

There is no checkpoint, layer, tensor-name, or original expert identifier in
the authorization, moment contract, generator key, or publication receipt.
Any private mapping into these canonical names is an external audit concern.

## External byte authorization

`AUTHORIZATION_TEMPLATE.json` fixes the complete order and geometry:

- ordinals are exactly slot-major, then `gate`, `up`, `down`;
- `gate` and `up` have storage shape `[768, 2048]`;
- `down` has storage shape `[2048, 768]`;
- every matrix has 1,572,864 BF16 values and 3,145,728 bytes;
- the total is 18 matrices, 28,311,552 values, and 56,623,104 bytes;
- filenames are canonical identity-free basenames;
- every row carries an externally computed SHA-256 of the exact BF16 bytes;
- the ordered rows have a domain-separated source-set root;
- the record binds the source artifact, both source geometries, source
  pipeline, score receipt, and independent moment-auditor source.

Replacing placeholders is not sufficient. The completed record has an
internal canonical-JSON seal, and the reviewed dispatcher must receive the
SHA-256 of the exact canonical-pretty authorization file through a separate
trusted channel. This prevents an attacker from substituting and self-sealing
a different source set. The authorization file is capped at 131,072 bytes;
the matrix count, per-file size, total bytes, and output member set are fixed.

Before computing any moment, the runtime authenticates the whole 18-file set.
It accepts only regular, non-symlink files under the canonical source root,
checks open-descriptor identity before and after the read, checks the exact
length, and hashes the same bytes later interpreted as BF16. No directory
enumeration or model-name inference is used.

## Frozen FP64 moments

Each payload is a headerless C-order stream of little-endian BF16 words in its
declared storage orientation. The pinned CPU operation is:

1. interpret the payload as little-endian `uint16`;
2. shift each word left 16 bits, view it as IEEE binary32, then exactly widen
   it to IEEE binary64;
3. flatten in C order;
4. compute `mean = np.mean(values, dtype=np.float64)`;
5. compute `centered = values - mean`, then
   `centered_sse = np.sum(centered * centered, dtype=np.float64)`;
6. separately compute `energy = np.sum(values * values, dtype=np.float64)`;
7. serialize each scalar as its exact little-endian binary64 bit pattern in
   lowercase hexadecimal.

All values must be finite, and both centered SSE and energy must be positive.
There is no `ddof` correction. `runtime_pins.json` freezes CPython 3.12.3,
NumPy 2.5.2, the Linux x86-64 interpreter bytes, NumPy origin and distribution
RECORD bytes, byte order, isolated launch flags, and the exact consumer source
files and manifest.

## Frozen Gaussian regeneration law

The reference implementation independently matches the producer law. For
each frozen global seed and moment row, SHA-256 derives one 128-bit seed from
the domain, global seed, ordinal, and public generator key. A fresh NumPy
PCG64 generator makes exactly one binary64 standard-normal vector. The vector
is centered and scaled in FP64. Six fixed affine iterations each round through
binary32 to BF16 with round-to-nearest, ties-to-even. The lowest preregistered
normalized objective wins, with the earliest iteration winning ties.

The selected matrix must satisfy:

```text
abs(control_mean - source_mean) / source_RMS <= 2^-17
abs(control_centered_RMS / source_centered_RMS - 1) <= 2^-15
```

No seed search, source-derived candidate choice, or identity-bearing key is
allowed.

## Fail-closed publication

Direct execution exits 3. A reviewed dispatcher imports
`publish_authenticated_contract` and must provide both out-of-band digests,
the pinned NumPy module, canonical source root, exact authorization bytes, and
an absent output path.

The API verifies package source, authorization, and runtime before any source
open. It authenticates all payload bytes before calculating any moment. Only
after all 18 rows and the producer ABI revalidate does it create a sibling
`.incomplete-*` staging directory. It writes and fsyncs:

```text
MOMENT_CONTRACT.json
PUBLICATION.json
RUNTIME_PINS.json
SOURCE_AUTHORIZATION.json
COMPLETE.json
```

The requested final directory appears only through a same-parent atomic
rename after `COMPLETE.json` exists. It is never overwritten. A write failure
leaves the requested final path absent and, at most, a clearly named forensic
staging directory. Source payloads are never copied into the publication.

Publication remains nonpromoting and grants no control-generation, encoding,
scoring, accelerator, or scientific-claim authority.

## Source-only verification

From the repository root:

```text
python -I -B research/uwfa_sc_v9_bf16_source_moment_contract_v0/test_source_only.py
python -I -B research/uwfa_sc_v9_bf16_source_moment_contract_v0/test_hostile.py
python -I -B research/uwfa_sc_v9_bf16_source_moment_contract_v0/verify_source.py \
  --package research/uwfa_sc_v9_bf16_source_moment_contract_v0 \
  --repository-root .
```

These commands use synthetic data only. They do not open, stat, hash, or
enumerate any production BF16 payload; they do not import accelerator modules;
and they do not execute the production publisher.

## Blockers

No actual `MOMENT_CONTRACT.json` can exist until an independent source auditor
provides the completed 18-row authorization and its out-of-band file digest.
The pinned production CPU tuple must then be independently revalidated. A
fresh independent source audit of this package and the separately reviewed v9
control consumer remain required before any payload access or experiment.
