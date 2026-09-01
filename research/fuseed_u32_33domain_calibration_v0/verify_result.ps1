$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root 'calibrate.py'
$resultPath = Join-Path $root 'result.json'
$result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Close-Enough([double]$Actual, [double]$Expected) {
    $scale = [Math]::Max(1.0, [Math]::Abs($Expected))
    return [Math]::Abs($Actual - $Expected) -le (1e-12 * $scale)
}

function Median-Two([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    return ($sorted[0] + $sorted[1]) / 2.0
}

$scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $scriptPath).Hash.ToLowerInvariant()
Require ($result.schema -eq 'fuseed_u32_source_free_33domain_calibration_v0') 'schema mismatch'
Require ($result.script_sha256 -eq $scriptHash) 'script hash mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0) 'model path access is nonzero'
Require ($result.access.payload_files_opened -eq 0) 'payload access is nonzero'
Require ($result.access.network_operations -eq 0) 'network access is nonzero'
Require ($result.runtime.cuda_visible_devices -eq '0') 'CUDA visibility mismatch'

$shape = $result.frozen_shape_emulated
Require ($shape.domains -eq 33) 'domain count mismatch'
Require ($shape.normal4_bundles_per_candidate -eq 256) 'bundle count mismatch'
Require ($shape.widened_bf16_anchor_values_per_candidate -eq 1024) 'anchor count mismatch'
Require ($shape.fit_counts.up -eq 244 -and $shape.fit_counts.down -eq 268) 'fit counts mismatch'
Require ($shape.score_counts.up -eq 244 -and $shape.score_counts.down -eq 268) 'score counts mismatch'
Require ($shape.common_fp64_moments -eq 8) 'common-moment count mismatch'
Require ($shape.domain_fp64_cross_moments -eq 4) 'cross-moment count mismatch'
Require ($shape.candidate_count -eq 16777216) 'candidate count mismatch'
Require ($shape.q_shape[0] -eq 33 -and $shape.q_shape[1] -eq 16777216) 'q shape mismatch'
$expectedQBytes = [uint64]33 * [uint64]16777216 * [uint64]4
Require ([uint64]$shape.q_bytes -eq $expectedQBytes) 'q byte count mismatch'
Require ($shape.top_k_per_domain -eq 8192) 'Top-K mismatch'
Require (@($result.rows).Count -eq 2) 'row count mismatch'

Require ($result.kernel.attributes.num_regs -eq 106) 'register count mismatch'
Require ($result.kernel.attributes.local_size_bytes -eq 0) 'local spill proxy is nonzero'
Require ($result.kernel.register_spill_proxy_local_size_bytes -eq 0) 'recorded spill proxy mismatch'
Require ($result.kernel.block_threads -eq 256 -and $result.kernel.warps_per_block -eq 8) 'launch shape mismatch'
Require ($result.kernel.grid_blocks -eq 65535) 'grid mismatch'
Require (@($result.rows.domain_topk_seed_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K seed hashes differ'
Require (@($result.rows.domain_topk_value_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K value hashes differ'
Require (@($result.rows.q_sentinel_sha256 | Select-Object -Unique).Count -eq 1) 'q sentinel hashes differ'
Require (@($result.rows.best_seed_domain_0 | Select-Object -Unique).Count -eq 1) 'best seeds differ'
Require (@($result.rows.best_q_domain_0 | Select-Object -Unique).Count -eq 1) 'best q differs'

foreach ($row in $result.rows) {
    Require (Close-Enough (16777216.0 / [double]$row.kernel_seconds) ([double]$row.candidate_rate_per_second_kernel)) 'candidate rate mismatch'
    Require (Close-Enough (16777216.0 * 1024.0 / [double]$row.kernel_seconds) ([double]$row.anchor_value_rate_per_second_kernel)) 'anchor rate mismatch'
    Require (Close-Enough (16777216.0 * 33.0 * 1024.0 / [double]$row.kernel_seconds) ([double]$row.domain_cross_moments_per_second_kernel)) 'cross-moment rate mismatch'
    Require ($row.max_boundary_tie_cardinality -ge 1) 'invalid tie cardinality'
}

$kernelMedian = Median-Two @($result.rows.kernel_seconds)
$e2eMedian = Median-Two @($result.rows.end_to_end_seconds_excluding_journal)
Require (Close-Enough $kernelMedian ([double]$result.aggregate.median_kernel_seconds_per_shard)) 'kernel median mismatch'
Require (Close-Enough $e2eMedian ([double]$result.aggregate.median_end_to_end_seconds_per_shard_excluding_journal)) 'end-to-end median mismatch'
Require ($result.aggregate.shards_per_abi -eq 256 -and $result.aggregate.abis -eq 3) 'search multiplicity mismatch'
$kernelProjection = $kernelMedian * 256.0 * 3.0
$e2eProjection = $e2eMedian * 256.0 * 3.0
Require (Close-Enough $kernelProjection ([double]$result.aggregate.projected_three_abi_kernel_seconds)) 'kernel projection mismatch'
Require (Close-Enough $e2eProjection ([double]$result.aggregate.projected_three_abi_end_to_end_seconds_excluding_journal)) 'end-to-end projection mismatch'
Require ($kernelProjection -gt 900.0) 'recorded surrogate projection changed unexpectedly'
Require ($result.aggregate.replay_deterministic -eq $true) 'determinism flag mismatch'

Write-Output 'PASS: surrogate result is internally consistent; it is not direct-counter-equivalent and has no FUSEED-v1 decision authority.'
