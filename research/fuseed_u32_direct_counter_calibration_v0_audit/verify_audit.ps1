$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
function Require([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Hash([string]$Path) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }

$receiptPath = Join-Path $root 'audit_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json -Depth 100
Require ($receipt.schema -ceq 'fuseed_u32_direct_counter_calibration_v0_independent_audit_receipt_v1') 'receipt schema'
Require ($receipt.status -ceq 'BLOCKED_FROZEN_V1_RELEASE_PREREQUISITES_UNMET') 'receipt status'
Require ($receipt.verdict -ceq 'BLOCK_DO_NOT_FINALIZE_FUSEED_V1_RUNTIME_KILL') 'receipt verdict'
Require (@($receipt.release_blockers).Count -eq 4) 'blocker count'
Require (($receipt.release_blockers.id -join '|') -ceq 'RUNTIME_COMPILER_AND_COMPILED_KERNEL_IDENTITY_UNBOUND|SECOND_REQUIRED_REFERENCE_CONSTRUCTION_AND_PARITY_REPETITIONS_ABSENT|TORCH_GENERATOR_STATE_PARITY_ABSENT|FINAL_PLAN_AND_JOURNAL_PERFORMANCE_SHAPE_NOT_BOUND') 'blocker identities/order'
Require ($receipt.authorization.calibration_release_passed -eq $false) 'release blocked'
Require ($receipt.authorization.fuseed_v1_runtime_kill_finalized -eq $false) 'v1 kill not finalized'
Require ($receipt.authorization.model_or_qwen_access_authorized -eq $false) 'model access blocked'
Require ($receipt.authorization.gpu_or_cupy_execution_authorized -eq $false) 'GPU access blocked'
Require ($receipt.access_ledger.model_or_qwen_files_opened_statted_hashed_enumerated_or_named -eq 0) 'zero model access'
Require ($receipt.access_ledger.cupy_imports -eq 0 -and $receipt.access_ledger.cuda_initializations -eq 0 -and $receipt.access_ledger.gpu_jobs -eq 0) 'zero CuPy/CUDA/GPU access'

$payload = @(
    $receipt.schema,
    $receipt.status,
    $receipt.audited_target.artifact_manifest_sha256,
    $receipt.audited_target.script_sha256,
    $receipt.audited_target.result_sha256,
    $receipt.audited_frozen_gate.pre_audit_design_sha256,
    @($receipt.release_blockers.id),
    $receipt.verdict
) -join "`n"
$bytes = [Text.Encoding]::UTF8.GetBytes($payload)
$digest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
Require ($receipt.audit_internal_sha256 -ceq $digest) 'receipt internal seal'

$manifestPath = Join-Path $root 'ARTIFACT_SHA256SUMS.txt'
Require (Test-Path -LiteralPath $manifestPath -PathType Leaf) 'audit manifest exists'
$expectedMembers = @('ARTIFACT_SHA256SUMS.txt','README.md','audit_receipt.json','run_source_audit.ps1','verify_audit.ps1')
$actualMembers = @(Get-ChildItem -LiteralPath $root -Force | ForEach-Object { $_.Name } | Sort-Object)
Require (($actualMembers -join '|') -ceq (($expectedMembers | Sort-Object) -join '|')) 'audit exact package closure'
$rows = @(Get-Content -LiteralPath $manifestPath | Where-Object { $_.Trim() })
Require ($rows.Count -eq 4) 'audit manifest row count'
foreach ($row in $rows) {
    Require ($row -match '^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$') 'audit manifest row grammar'
    Require ($Matches[2] -ne 'ARTIFACT_SHA256SUMS.txt') 'manifest excludes itself'
    Require ((Hash (Join-Path $root $Matches[2])) -ceq $Matches[1]) "audit member hash $($Matches[2])"
}

Write-Output "PASS: audit package is hash-closed; verdict remains BLOCK_DO_NOT_FINALIZE_FUSEED_V1_RUNTIME_KILL."
