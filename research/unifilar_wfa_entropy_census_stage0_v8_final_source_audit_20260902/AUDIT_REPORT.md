# UWFA-SC v8 final independent sealed-source review

Date: 2026-09-02  
Reviewed commit: `d563c4ac1e78a6b6e7f0722291211d1209f775af`  
Reviewed parent: `2315551e504b0c7c1e357793aa259b745ff4d717`

## Verdict

**PASS_INDEPENDENT_SOURCE_REVIEW**

The exact sealed source at commit `d563c4a` is the implementation that passed
the earlier pre-freeze audit. The freeze transition changed only lifecycle
prose in `README.md`, the single `design_lock.json.status` value, and added the
literal `SOURCE_MANIFEST.json`. Every Python member and every other
non-lifecycle member is byte-identical to the independently accepted parent.

The literal source manifest is 3,518 bytes with SHA-256:

`a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6`

All 17 declared members, 633,319 bytes excluding the manifest, match their
exact manifest sizes and SHA-256s. The complete 18-file sealed package is
636,837 bytes. See `SOURCE_INVENTORY.tsv`.

This is a source-only verdict. It does not authorize or report any Qwen,
current-artifact, Gaussian-control, entropy, rate, MSE, cold-read, or runtime
performance claim.

## Independent authentication

`independent_source_review.py` imports no producer module. Through held regular
file reads and strict duplicate-key JSON parsing, it independently verifies:

- the exact 40-character commit and its sole exact parent;
- a clean tracked checkout;
- the exact three-path freeze transition;
- byte equality of all protected pre-freeze and sealed Git blobs;
- the exact allowed `design_lock` status transition and no other design change;
- the exact pre-freeze and sealed README hashes;
- the exact package member set, literal manifest bytes, row order, member sizes,
  member hashes, committed-blob equality, and Python syntax.

The final structured receipt is `INDEPENDENT_SOURCE_REVIEW.json`, with schema
`unifilar-wfa-entropy-census-independent-source-review-v8` and status
`PASS_INDEPENDENT_SOURCE_REVIEW`.

## Isolated RunPod execution

The audit used an isolated detached sparse checkout at:

`/workspace/uwfa_v8_final_source_audit_d563c4a_20260902/clean-repo`

Only `research/unifilar_wfa_entropy_census_stage0_v8` was checked out. The
RunPod could not anonymously clone the GitHub URL because the repository
requested credentials. Therefore a complete-history Git bundle was generated
from the already-pushed local origin, SHA-256 authenticated on both hosts,
verified with `git bundle verify`, and cloned into the clean sparse checkout.
The bundle was 51,886,783 bytes with SHA-256:

`e4a525cf6b8ffe71c30573e0dab101c36c31af77d27e5c3b449a1be5701c0988`

The checkout resolved exactly to `d563c4a`, retained the exact parent, and was
tracked-clean before and after verification and tests.

Environment:

- host: `1113c8d18a24`;
- Python: `/usr/bin/python3.12`, Python 3.12.3;
- Python executable SHA-256:
  `1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5`;
- GPU present: NVIDIA GeForce RTX 5090,
  `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722`, PCI
  `00000000:16:00.0`, driver 580.126.09.

The GPU identity records the requested RunPod environment. This review did not
claim a fresh all-150 CUDA performance replay; that remains a separate gate.

## Sealed verifier and test results

The package verifier ran in isolated mode:

```text
/usr/bin/python3.12 -I -B verify_source.py \
  --package /workspace/uwfa_v8_final_source_audit_d563c4a_20260902/clean-repo/research/unifilar_wfa_entropy_census_stage0_v8 \
  --compact
```

It returned `PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY`, bound the same
manifest SHA-256, reported all 150 candidate cells and the universal 1–256
expert/32-byte owner-set ABI, and granted no payload authority. Exit code was
zero; stderr was empty.

The complete POSIX suite then ran from the same clean checkout:

```text
/usr/bin/python3.12 -I -B test_source_only.py
```

Result: **68 tests run, 68 explicit `ok` lines, zero failures, zero errors, zero
skips, `OK`**, in 78.303 reported seconds. The exact captured unittest stderr
was 10,804 bytes with SHA-256
`08a25d0d5bc263e1b3c095b49e511ef29efd5a5df56a7ddf04c39831258497d2`.

The sealed manifest test is no longer the pre-freeze expected skip; it ran and
passed. A prior direct run in the same clean checkout also passed 68/68 in
77.427 seconds, giving a separate consistent replay.

## Claim boundary and next gates

No Qwen/model payload, current finite artifact or selected stream, or Gaussian
control was opened, statted, hashed, or enumerated. No producer source was
edited. This review completes the independently pinned sealed-source check only.

Before any payload opening, the protocol still requires an independently
audited pinned dispatcher/bootstrap. After that it requires a fresh exact-public-
commit RTX 5090 all-150 and representative source-free replay and a fresh-process
independent audit of every eventual payload result and emitted byte. None of
those later gates is implied by this PASS.
