# CBIB-1 Qwen layer-15 deployment successor r3

This is a separately named, source-closed successor to the deterministically
blocked r2 snapshot. Earlier snapshots and their independent reviews remain
unchanged. This package repairs only the r2 source-free fixture/read-survivor
contract, retains its evaluator-schema and NumPy closure repairs, and otherwise retains the universal,
already-quantized-label CBIB-1 core, CuPy worker, and authenticated panel.

Current status: `HOLD_PENDING_NEW_INDEPENDENT_DEPLOYMENT_REVIEW`. It has not
been staged on RunPod, has not read Qwen payloads, and has no production
authority.

## Repair 1: complete source-free CPU/CuPy parity

`run_source_free_cupy.py` uses a deterministic 16-expert, two-role fixture
with 131,072 coordinates per role. All eight production 2,048-coordinate folds
are populated. It directly checks all group sizes `2,4,8,16`; every
role/group/fold hard-EM model; canonical-model training and held-out
assignments; independently reconstructed, vectorized NumPy latent/conditional
counts for both; pairwise MI; partitions; the entire integer rate/read ledger;
status; and every floating field with an explicit bound. A small executable
regression covers the frozen CPU evaluator's actual assignments-only schema.

The fixture has a balanced latent, a frozen 0.105 sign-noise grid point, and
role-reflected pair ownership. Its scale ledger is exactly 256 bytes per expert:
two roles times 64 blocks times two bytes. The cheap targeted group-size-2
regression passes at 5/2 bpw with exact maximum amplification
1.9651249492746525. The full preflight still requires all group sizes, folds,
models and controls. It requires at least one favorable source candidate to survive an exact
strict-read endpoint, making all eight affine controls actually execute.
`EXPECTED_PREFLIGHT.json` freezes the fixture hash and acceptance contract.
A preflight PASS remains source-free runtime evidence, not Qwen evidence.

## Repair 2: NumPy source/native binary closure before claim

The launcher pins NumPy 2.5.2, its resolved entry path and hash, and the exact
wheel RECORD. Before the one-use claim, it authenticates every RECORD-hashed
member under `numpy/`, `numpy.libs/`, and the distribution metadata,
including the extension modules and bundled OpenBLAS objects used by
decision-affecting dot/log2/dtype operations. Only RECORD itself may be
unhashed. Python, CuPy, CUDA, device, package, panel, payload/output paths and
freshness are still checked before the claim.

## Source-only verification

```powershell
$pkg = "research/same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3"
$sha = (Get-FileHash -Algorithm SHA256 "$pkg/SOURCE_MANIFEST.json").Hash.ToLowerInvariant()
python -I -B "$pkg/verify_source.py" --package $pkg --manifest-sha256 $sha
python -I -B "$pkg/test_source_only.py"
```

After a new independent review, the separately authorized source-free runtime
preflight is:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/int2-cupy-venv/bin/python -I -B \
  run_source_free_cupy.py \
  --authorization RUN_SOURCE_FREE_CBIB1_QWEN_DEPLOYMENT_PARITY_V0_R3 \
  --deployment-manifest-sha256 <independently-pinned-sha256>
```

No Qwen invocation is authorized by this README. A future reviewer must
explicitly authorize the single fixed-path production launch.
