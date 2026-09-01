$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root 'calibrate_direct.py'
$resultPath = Join-Path $root 'result.json'
$shapePath = Join-Path (Split-Path -Parent $root) 'fuseed_u32_33domain_calibration_v0\calibrate.py'
$result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Close-Enough([double]$Actual, [double]$Expected) {
    $scale = [Math]::Max(1.0, [Math]::Abs($Expected))
    return [Math]::Abs($Actual - $Expected) -le (1e-12 * $scale)
}

$scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $scriptPath).Hash.ToLowerInvariant()
$shapeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $shapePath).Hash.ToLowerInvariant()
Require ($result.schema -eq 'fuseed_u32_source_free_direct_counter_calibration_v0') 'schema mismatch'
Require ($result.script_sha256 -eq $scriptHash) 'script hash mismatch'
Require ($result.shape_source.sha256 -eq $shapeHash) 'shape source hash mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0) 'model path access is nonzero'
Require ($result.access.payload_files_opened -eq 0) 'payload access is nonzero'
Require ($result.access.network_operations -eq 0) 'network access is nonzero'
Require ($result.runtime.python -eq '3.12.3') 'Python version mismatch'
Require ($result.runtime.numpy -eq '2.5.2') 'NumPy version mismatch'
Require ($result.runtime.cupy -eq '14.2.0') 'CuPy version mismatch'
Require ($result.runtime.cuda_runtime -eq 12090) 'CUDA runtime mismatch'
Require ($result.runtime.device -eq 'NVIDIA GeForce RTX 5090') 'device mismatch'
Require ($result.runtime.cuda_visible_devices -eq '0') 'CUDA visibility mismatch'

$expectedHeaders = @{
    '/usr/local/cuda/include/curand_normal.h' = '967998564d9f9f4a045563b2b5d2a15eb1cbdfa18b0a332707c3a765e09a61c0'
    '/usr/local/cuda/include/curand_kernel.h' = '4a37c07a1d77c9b5c8c627a4720733cee6b4da4200a844e8a49291858bc26adf'
    '/usr/local/cuda/include/curand_philox4x32_x.h' = '4f6d483fe45d837fed49553d25ee1d2cabb012a138a2f7f08bbaf584e63dd83c'
    '/usr/local/cuda/include/cuda_bf16.h' = '4d5a2ad88adb17983aef0505ed6ed2a0603497c79c103bba82c928301ea12310'
    '/usr/local/cuda/include/cuda_fp16.h' = '8eb1600a8e2e33d40572bffe7001a27f6046a74949d80353fe02cf88b2563dda'
}
foreach ($entry in $expectedHeaders.GetEnumerator()) {
    Require ($result.cuda_headers.($entry.Key) -eq $entry.Value) "header binding mismatch: $($entry.Key)"
}

Require ($result.direct_source.curand_init_calls_in_performance_kernel -eq 0) 'performance kernel calls curand_init'
Require ($result.direct_source.curand_normal4_calls_in_performance_kernel -eq 0) 'performance kernel calls curand_normal4'
Require ($result.direct_source.direct_philox_calls_in_performance_kernel -eq 1) 'direct Philox call count mismatch'
Require ($result.direct_source.box_muller_pair_calls_in_performance_kernel -eq 2) 'Box-Muller call count mismatch'
Require ($result.direct_source.derived_cuda_sha256 -eq '8665346ef3319787f3cf95edc2b2fea29f24242d46745b79b17dc3a9975aee46') 'derived CUDA hash mismatch'

$parity = $result.parity
Require ($parity.rows -eq 132) 'parity row count mismatch'
Require ($parity.raw_bitwise_equal -eq $true) 'raw parity failed'
Require ($parity.scaled_bf16_bitwise_equal -eq $true) 'BF16 parity failed'
Require ($parity.terminal_counter_equal -eq $true) 'terminal counter parity failed'
Require (($parity.zero_kat_words -join ' ') -eq '6627e8d5 e169c58d bc57ac4c 9b00dbd8') 'zero KAT mismatch'
Require ($parity.vector_sha256 -eq '3bcb66a01866882c05d67026869f8cd258631cbcbab4642846f7c3ae640349ba') 'parity vector hash mismatch'
Require ($parity.raw_float32_sha256 -eq '58601f456e20857871cb012ed979a8cd930e790d79ecf20b9911e3fde4c3f27c') 'raw parity hash mismatch'
Require ($parity.scaled_widened_bf16_sha256 -eq '3dfafb5b177db96730ea9f12214bf8925bc4af2c10e3b9e79fb0d3963f951366') 'scaled parity hash mismatch'
Require ($parity.terminal_counter_sha256 -eq '0b5397649cd8f389c8db9675b92df3dafe08b34b8ef0168eed2d0d929112de1e') 'terminal hash mismatch'

$shape = $result.shape
Require ($shape.candidates -eq 16777216 -and $shape.domains -eq 33) 'candidate/domain shape mismatch'
Require ($shape.normal4_bundles_per_candidate -eq 256 -and $shape.normal_values_per_candidate -eq 1024) 'generator shape mismatch'
Require ($shape.fit_counts.up -eq 244 -and $shape.fit_counts.down -eq 268) 'fit counts mismatch'
Require ($shape.score_counts.up -eq 244 -and $shape.score_counts.down -eq 268) 'score counts mismatch'
Require ($shape.q_shape[0] -eq 33 -and $shape.q_shape[1] -eq 16777216) 'q shape mismatch'
$expectedQBytes = [uint64]33 * [uint64]16777216 * [uint64]4
Require ([uint64]$shape.q_bytes -eq $expectedQBytes) 'q bytes mismatch'
Require ($shape.top_k_per_domain -eq 8192 -and $shape.repetitions -eq 3) 'Top-K/repetition mismatch'
Require ($shape.scale_bits.up -eq '3c03126f' -and $shape.scale_bits.down -eq '3a560a28') 'scale bits mismatch'
Require ($result.kernel.attributes.num_regs -eq 108) 'register count mismatch'
Require ($result.kernel.attributes.local_size_bytes -eq 0) 'local spill proxy is nonzero'
Require ($result.kernel.grid_blocks -eq 65535 -and $result.kernel.block_threads -eq 256) 'launch shape mismatch'
Require (@($result.rows).Count -eq 3) 'row count mismatch'
Require (@($result.rows.domain_topk_seed_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K seed hashes differ'
Require (@($result.rows.domain_topk_value_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K value hashes differ'
Require (@($result.rows.q_sentinel_sha256 | Select-Object -Unique).Count -eq 1) 'q hashes differ'

foreach ($row in $result.rows) {
    Require (Close-Enough (16777216.0 / [double]$row.kernel_seconds) ([double]$row.candidate_rate_per_second_kernel)) 'candidate rate mismatch'
    Require (Close-Enough (16777216.0 * 256.0 / [double]$row.kernel_seconds) ([double]$row.normal4_bundle_rate_per_second_kernel)) 'bundle rate mismatch'
    Require (Close-Enough (16777216.0 * 1024.0 / [double]$row.kernel_seconds) ([double]$row.normal_value_rate_per_second_kernel)) 'normal rate mismatch'
    Require ($row.max_boundary_tie_cardinality -ge 1) 'invalid tie count'
}

$kernelMedian = [double](@($result.rows.kernel_seconds | Sort-Object)[1])
$selectionMedian = [double](@($result.rows.selection_seconds_total | Sort-Object)[1])
$coldExcess = [Math]::Max(0.0, [double]$result.rows[0].selection_seconds_total - $selectionMedian)
Require (Close-Enough $kernelMedian ([double]$result.aggregate.median_kernel_seconds_per_shard)) 'kernel median mismatch'
Require (Close-Enough $selectionMedian ([double]$result.aggregate.median_warm_selection_seconds_per_shard)) 'selection median mismatch'
Require (Close-Enough $coldExcess ([double]$result.aggregate.one_time_cold_selection_excess_seconds)) 'cold excess mismatch'
Require ($result.aggregate.shards_per_abi -eq 256 -and $result.aggregate.abis -eq 3) 'search multiplicity mismatch'
$kernelProjection = $kernelMedian * 256.0 * 3.0
$warmProjection = ($kernelMedian + $selectionMedian) * 256.0 * 3.0 + $coldExcess
Require (Close-Enough $kernelProjection ([double]$result.aggregate.projected_three_abi_kernel_seconds)) 'kernel projection mismatch'
Require (Close-Enough $warmProjection ([double]$result.aggregate.projected_three_abi_warm_end_to_end_seconds_excluding_journal)) 'warm projection mismatch'
Require ($result.aggregate.prospective_runtime_gate_seconds -eq 900.0) 'runtime gate mismatch'
Require ($kernelProjection -gt 900.0 -and $warmProjection -gt 900.0) 'prospective runtime kill did not trigger'
Require ($result.aggregate.kernel_projection_below_gate -eq $false) 'kernel gate flag mismatch'
Require ($result.aggregate.warm_e2e_projection_below_gate -eq $false) 'warm gate flag mismatch'
Require ($result.aggregate.replay_deterministic -eq $true) 'determinism flag mismatch'

Write-Output 'PASS: exact direct-counter calibration is internally consistent and prospectively kills FUSEED-v1 runtime before Qwen access, pending independent audit.'
