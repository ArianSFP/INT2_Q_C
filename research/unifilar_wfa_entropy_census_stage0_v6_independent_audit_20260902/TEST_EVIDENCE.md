# Independent source-only execution evidence

Date: 2026-09-02

This is audit evidence, not a producer manifest or freeze. No Qwen/model payload
path was opened. No payload authority was granted.

## Pre-execution authentication

Audited local tree:

`research/unifilar_wfa_entropy_census_stage0_v6`

Authenticated RunPod replay tree:

`/workspace/uwfa_v6_independent_audit_b14bf19a`

The local tree and the RunPod copy both had the exact 17-member inventory in
`SOURCE_INVENTORY.tsv`, totalling 572,546 bytes. The independent root was
computed over UTF-8 bytewise-ordinal member names using:

```
"UWFA-SC-V6-INDEPENDENT-INVENTORY-v1\0"
+ name + "\0" + decimal_bytes + "\0" + lowercase_sha256 + "\n"
```

The result on both hosts, before any authenticated Python source was compiled,
was:

`b14bf19aa8965f0ab22ec26db43cddd63e0c5f3c4d996edeed45e512e516cca2`

The RunPod path and every ancestor were inspected with `namei`; all were real
directories, not symbolic links. Exact membership, regular-file type, byte
length and SHA-256 were then checked through a held directory descriptor. Only
after those checks did the independent harness compile the authenticated
snapshots of `uwfa_common.py` and `result_envelope.py`.

The producer's own canonical-JSON source root in the authenticated GPU receipt
is `ba05b2e456898867f240aae661fd783a7bb7e414d0893c9055832c04818bca01`.
It differs from the independent audit root because the two roots deliberately
use different domains/serializations; the underlying 17 byte strings were
identical.

Independent hostile harness SHA-256 before RunPod execution:

`0c70bebfe2fc9a549330162d35cbe4057aeec050c21325df1d13f7f67fca0acf`

## Full authenticated source suite

RunPod command:

```
cd /workspace/uwfa_v6_independent_audit_b14bf19a
/usr/bin/python3.12 -I -B test_source_only.py
```

Observed result:

- 57 tests collected;
- 56 passed;
- 1 expected pre-manifest skip (`test_manifest_verifier_only_after_manifest_exists`);
- 0 failed and 0 errored;
- elapsed time 75.554 seconds.

The executed suite included all authenticated `test_*` methods, including the
source/control closure tests, all-eight-control replay tests, exact ordered
150-cell preflight rejection tests, literal container round-trip/rate/routed
read tests, explicit modeled-symbol-density and repeated-requested-byte tests,
one-layer/nested fold tests, resource admission tests, UUID/PCI helper tests,
triplet commitment tests, 128/250/256 expert portability tests, and the three
directory-entry publication fault windows.

This green suite is recorded as regression evidence only. The independent
static and adversarial audit in `AUDIT_REPORT.md` found properties that the
suite does not test.

## Independent publication adversary

RunPod command:

```
/usr/bin/python3.12 -I -B /workspace/uwfa_v6_hostile_audit_0c70bebf.py \
  --package /workspace/uwfa_v6_independent_audit_b14bf19a \
  --gpu-output-parent /workspace/uwfa_v6_gpu_parent_b14bf19a \
  --gpu-final-name gpu-receipt
```

Observed status:

`PASS_INDEPENDENT_SOURCE_ONLY_HOSTILE_AUDIT`

The harness independently established:

- a successful final directory was the same `(st_dev, st_ino)` as the retained
  staging directory;
- producer and independent exact-member rehashes agreed on directory root
  `279a51db4b082066271023d0c2f9037f6565cd1ff6a875983b480dc094490052`
  for the synthetic success transaction;
- a substitution before the named move was rejected;
- a substitution after the move but before marker publication was rejected;
- a substitution after marker linking made both producer completion and the
  independent consumer fail closed;
- `COMPLETE.json` without the separate parent marker was rejected;
- mutation of a committed member was rejected by exact member/root rehash;
- substitution of the marker name was rejected;
- the available `/proc/self/fd` fallback linked the held marker inode, not a
  replaced mutable anchor. In the recorded run both held and linked marker were
  device 51, inode 6467482622, and `mutable_anchor_not_authority=true`.

The synthetic transaction identifiers/inodes are ephemeral. They are evidence
of checks, not stable artifact identity.

## Authenticated RTX 5090 source-free replay

The replay emitted no production or payload claim:

- status: `PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY`;
- `payload_authority_granted=false`;
- `public_commit_evidence=false`;
- no Qwen/model payload was opened.

Independent GPU identity in the receipt:

- name: `NVIDIA GeForce RTX 5090`;
- UUID: `GPU-c06e0fe0-9836-2f98-8f10-0514d085f722`;
- PCI bus: `00000000:16:00.0`;
- identity receipt SHA-256:
  `460816b0d11ce88c60a35bc3d9152e97f6bbe3a4a39a2fbd9165ec3181eac6c1`.

The independently parsed authenticated receipt contained exactly 150 cell
records. Their selector ordinals were unique and exactly `0..149`; declared and
actual count were both 150. Relevant authenticated fields were:

- all-150 status: `PASS_ALL_150_CPU_CUPY_EXACT_REPEATED`;
- candidate-selector SHA-256:
  `1a7b80b181100aba628ccce7ba02bab13462893ae11015f7d9e72184bacbfeca`;
- cell-results SHA-256:
  `ad045038391c5742e521e5a299b531226563ad37e605f4fd8797a69221b2e437`;
- all-150 elapsed: 14.5313749930356 seconds;
- representative status: `PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD`;
- representative candidate count: 150, all scored;
- representative winner: suffix, 2 states, reset 4096, selector ordinal 4;
- representative measured updates: 224,366,256;
- measured time: 28.1783462989843 seconds;
- projected time: 4,337.43892085413 seconds against a 21,600 second cap;
- recorded representative phase totals: 304 kernels, 182,736,318 count
  updates, 41,629,938 length updates, 9,638,224 H2D bytes, and 19,380,720
  D2H bytes.

The receipt directory was independently opened and every declared regular
member was rehashed through a held directory descriptor. Exact committed
identity:

- final directory: device 66307, inode 4334899031;
- parent marker: device 66307, inode 8612513008;
- receipt directory root:
  `49f9151b6fc57ff925cef6eac2b876c69f81f9f1eb5c05d04c8c89a0edc34d36`;
- parent commit:
  `306cff2bdc3bb933fd7fe98f69f7412206a0c768272ad47f16fee5126beb60ce`;
- parent-marker file SHA-256:
  `527abfde8e92ffdc122e278c781154da0c81ce723818b5738ccc4cf1443fa1b0`;
- `GPU_DEV_RECEIPT.json` SHA-256:
  `8bfb94a9240893019aeaaf981bac126ded608e6f18efb8cd049f963bd12e628b`;
- `RUN_STATE.json` SHA-256:
  `702e967bf2d01c42956fea8d448bca7cf115f86f9c1dcb7d1be72b2f26826b6c`;
- `COMPLETE.json` SHA-256:
  `95c1cf23067743267c07a691e5cee6d0f70dab3c6ab8e7fe127612743aa4719c`.

Authenticated copies of those four JSON byte strings are retained beside this
file. Copying them does not preserve RunPod inode authority; the independent
RunPod held-descriptor verification above supplies that evidence.
