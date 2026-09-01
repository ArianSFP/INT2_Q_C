$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root 'benchmark.py'
$resultPath = Join-Path $root 'result.json'
$result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Close-Enough([double]$Actual, [double]$Expected) {
    $scale = [Math]::Max(1.0, [Math]::Abs($Expected))
    return [Math]::Abs($Actual - $Expected) -le (1e-12 * $scale)
}

$scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $scriptPath).Hash.ToLowerInvariant()
Require ($result.schema -eq 'fuseed_u32_source_free_throughput_probe_v0') 'schema mismatch'
Require ($result.script_sha256 -eq $scriptHash) 'script hash mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0) 'model path access is nonzero'
Require ($result.access.payload_files_opened -eq 0) 'payload access is nonzero'
Require ($result.access.network_operations -eq 0) 'network access is nonzero'
Require ($result.runtime.cuda_visible_devices -eq '0') 'CUDA visibility mismatch'
Require ($result.shape.candidates -eq 16777216) 'candidate count mismatch'
Require ($result.shape.normal4_bundles_per_candidate -eq 80) 'bundle count mismatch'
Require ($result.shape.normal_values_per_candidate -eq 320) 'normal count mismatch'
Require ($result.shape.top_k -eq 2048) 'Top-K mismatch'
Require (@($result.rows).Count -eq 3) 'row count mismatch'
Require (@($result.rows.topk_seed_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K seed hashes differ'
Require (@($result.rows.topk_value_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K value hashes differ'
Require (@($result.rows.best_seed | Select-Object -Unique).Count -eq 1) 'best seeds differ'
Require (@($result.rows.best_score | Select-Object -Unique).Count -eq 1) 'best scores differ'

$kernelMedian = [double](@($result.rows.kernel_seconds | Sort-Object)[1])
$endMedian = [double](@($result.rows.end_to_end_seconds | Sort-Object)[1])
$projectionScale = [Math]::Pow(2.0, 32) / [double]$result.shape.candidates
Require (Close-Enough $kernelMedian ([double]$result.aggregate.median_kernel_seconds)) 'kernel median mismatch'
Require (Close-Enough $endMedian ([double]$result.aggregate.median_end_to_end_seconds)) 'end-to-end median mismatch'
Require (Close-Enough ($kernelMedian * $projectionScale) ([double]$result.aggregate.projected_full_u32_kernel_seconds_linear)) 'kernel projection mismatch'
Require (Close-Enough ($endMedian * $projectionScale) ([double]$result.aggregate.projected_full_u32_end_to_end_seconds_linear)) 'end-to-end projection mismatch'

Write-Output 'PASS: source-free throughput result is internally consistent and repeat deterministic.'
