$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root 'complete_scan.py'
$kernelPath = Join-Path (Split-Path -Parent $root) 'fuseed_u32_throughput_probe_v0\benchmark.py'
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
$kernelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $kernelPath).Hash.ToLowerInvariant()
Require ($result.schema -eq 'fuseed_u32_source_free_complete_scan_probe_v0') 'schema mismatch'
Require ($result.script_sha256 -eq $scriptHash) 'script hash mismatch'
Require ($result.kernel_source.sha256 -eq $kernelHash) 'kernel source hash mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0) 'model path access is nonzero'
Require ($result.access.payload_files_opened -eq 0) 'payload access is nonzero'
Require ($result.access.network_operations -eq 0) 'network access is nonzero'
Require ($result.runtime.cuda_visible_devices -eq '0') 'CUDA visibility mismatch'
Require ($result.shape.candidate_domain -eq 'inclusive uint32 0..4294967295') 'domain label mismatch'
Require ([uint64]$result.shape.candidate_count -eq [uint64]4294967296) 'candidate count mismatch'
Require ([uint64]$result.shape.shard_candidates * [uint64]$result.shape.shard_count -eq [uint64]4294967296) 'shard coverage mismatch'
Require ($result.shape.shard_count -eq 256) 'shard count mismatch'
Require ($result.shape.normal4_bundles_per_candidate -eq 80) 'bundle count mismatch'
Require ($result.shape.normal_values_per_candidate -eq 320) 'normal count mismatch'
Require ($result.shape.top_k -eq 2048) 'Top-K mismatch'
Require ($result.shape.tie_order -eq 'score descending, uint32 seed ascending') 'tie order mismatch'
Require (@($result.rows).Count -eq 2) 'row count mismatch'
Require (@($result.rows.global_topk_seed_sha256 | Select-Object -Unique).Count -eq 1) 'global Top-K seed hashes differ'
Require (@($result.rows.global_topk_value_sha256 | Select-Object -Unique).Count -eq 1) 'global Top-K value hashes differ'
Require (@($result.rows.per_shard_topk_sha256 | Select-Object -Unique).Count -eq 1) 'per-shard Top-K hashes differ'
Require (@($result.rows.best_seed | Select-Object -Unique).Count -eq 1) 'best seeds differ'
Require (@($result.rows.best_score | Select-Object -Unique).Count -eq 1) 'best scores differ'

foreach ($row in $result.rows) {
    Require ([uint64]$row.full_domain_first_seed -eq [uint64]0) 'first seed mismatch'
    Require ([uint64]$row.full_domain_last_seed -eq [uint64]4294967295) 'last seed mismatch'
    Require ([uint64]$row.full_domain_candidate_count -eq [uint64]4294967296) 'row count mismatch'
    Require ($row.shard_count -eq 256) 'row shard count mismatch'
    Require ($row.max_boundary_tie_cardinality -ge 1) 'invalid tie cardinality'
    Require (Close-Enough ([uint64]4294967296 / [double]$row.wall_seconds) ([double]$row.candidate_rate_per_second_wall)) 'candidate rate mismatch'
    Require (Close-Enough ([uint64]4294967296 * 320.0 / [double]$row.kernel_seconds_sum) ([double]$row.normal_value_rate_per_second_kernel)) 'normal rate mismatch'
}

function Median-Two([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    return ($sorted[0] + $sorted[1]) / 2.0
}

$wallMedian = Median-Two @($result.rows.wall_seconds)
$kernelMedian = Median-Two @($result.rows.kernel_seconds_sum)
$selectionMedian = Median-Two @($result.rows.selection_seconds_sum)
$mergeMedian = Median-Two @($result.rows.host_merge_seconds_sum)
Require (Close-Enough $wallMedian ([double]$result.aggregate.median_full_scan_wall_seconds)) 'wall median mismatch'
Require (Close-Enough $kernelMedian ([double]$result.aggregate.median_kernel_seconds_sum)) 'kernel median mismatch'
Require (Close-Enough $selectionMedian ([double]$result.aggregate.median_selection_seconds_sum)) 'selection median mismatch'
Require (Close-Enough $mergeMedian ([double]$result.aggregate.median_host_merge_seconds_sum)) 'merge median mismatch'
Require ($result.aggregate.complete_replay_deterministic -eq $true) 'determinism flag mismatch'
Require ($result.aggregate.projection_used -eq $false) 'projection flag mismatch'

Write-Output 'PASS: exact full-u32 coverage and replay-deterministic Top-K verified.'
