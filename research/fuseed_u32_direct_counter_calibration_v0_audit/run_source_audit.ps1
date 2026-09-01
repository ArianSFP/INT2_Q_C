$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$script:Checks = 0
function Require([bool]$Condition, [string]$Message) {
    $script:Checks++
    if (-not $Condition) { throw $Message }
}
function Eq($Actual, $Expected, [string]$Message) {
    Require ($Actual -ceq $Expected) "$Message (actual=$Actual expected=$Expected)"
}
function Close([double]$Actual, [double]$Expected, [double]$Tolerance, [string]$Message) {
    Require ([Math]::Abs($Actual - $Expected) -le $Tolerance) "$Message (actual=$Actual expected=$Expected)"
}
function Hash([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = [System.IO.Path]::GetFullPath((Join-Path $root '..\fuseed_u32_direct_counter_calibration_v0'))
$expectedMembers = @('README.md','artifact_sha256.txt','calibrate_direct.py','result.json','verify_result.ps1')
$actualMembers = @(Get-ChildItem -LiteralPath $target -Force | ForEach-Object { $_.Name } | Sort-Object)
Eq ($actualMembers -join '|') (($expectedMembers | Sort-Object) -join '|') 'target exact member closure'
foreach ($name in $expectedMembers) {
    $item = Get-Item -LiteralPath (Join-Path $target $name) -Force
    Require (-not $item.PSIsContainer) "target member is a file: $name"
    Require (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "target member is not a reparse point: $name"
}

$expectedHashes = [ordered]@{
    'calibrate_direct.py' = 'f5a7c8b9a525e02d469ca974f9a6607030b2ca2822b66d4bce31604251516ed5'
    'README.md' = '00b3697cf7838a3a0dea0886eda7bf1792f82b8d97e5122ef2eafea68b159e1e'
    'result.json' = '9fcd38bc2aaf27ccdc82175c09d5e35015015d278e340e16c10d82580306d83f'
    'verify_result.ps1' = 'c34d72b913501f14d89b5161ff7eb1d0e887e37db143c34acb09726762a20356'
    'artifact_sha256.txt' = 'c4ddddcb32fa932abc164c617333d1267ef137c2d06b577b892f72f5497f6ab8'
}
foreach ($entry in $expectedHashes.GetEnumerator()) {
    Eq (Hash (Join-Path $target $entry.Key)) $entry.Value "target hash $($entry.Key)"
}

$manifestRows = @(Get-Content -LiteralPath (Join-Path $target 'artifact_sha256.txt') | Where-Object { $_.Trim() })
Eq $manifestRows.Count 4 'target manifest row count'
foreach ($row in $manifestRows) {
    Require ($row -match '^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$') 'target manifest row grammar'
    $name = $Matches[2]
    Require ($name -ne 'artifact_sha256.txt') 'target manifest excludes itself'
    Eq (Hash (Join-Path $target $name)) $Matches[1] "target manifest member $name"
}

$producerOutput = @(& pwsh -NoLogo -NoProfile -File (Join-Path $target 'verify_result.ps1'))
Require ($LASTEXITCODE -eq 0) 'producer verifier exit code'
Require (($producerOutput -join "`n") -match '^PASS: exact direct-counter calibration') 'producer verifier PASS output'

$resultText = Get-Content -Raw -LiteralPath (Join-Path $target 'result.json')
$result = $resultText | ConvertFrom-Json -Depth 100
$source = Get-Content -Raw -LiteralPath (Join-Path $target 'calibrate_direct.py')
Eq $result.schema 'fuseed_u32_source_free_direct_counter_calibration_v0' 'result schema'
Eq ([int]$result.parity.rows) 132 'parity rows'
Eq ([bool]$result.parity.raw_bitwise_equal) $true 'raw parity'
Eq ([bool]$result.parity.scaled_bf16_bitwise_equal) $true 'BF16 parity'
Eq ([bool]$result.parity.terminal_counter_equal) $true 'terminal counter parity'
Eq ([int]$result.shape.repetitions) 3 'timed score repetitions'
Eq (@($result.rows.q_sentinel_sha256 | Select-Object -Unique).Count) 1 'deterministic q sentinel'
Eq (@($result.rows.domain_topk_seed_sha256 | Select-Object -Unique).Count) 1 'deterministic TopK seed hash'
Eq (@($result.rows.domain_topk_value_sha256 | Select-Object -Unique).Count) 1 'deterministic TopK value hash'

$kernelMedian = [double](@($result.rows.kernel_seconds | Sort-Object)[1])
$selectionMedian = [double](@($result.rows.selection_seconds_total | Sort-Object)[1])
$coldExcess = [Math]::Max(0.0, [double]$result.rows[0].selection_seconds_total - $selectionMedian)
$kernelProjection = $kernelMedian * 768.0
$warmProjection = ($kernelMedian + $selectionMedian) * 768.0 + $coldExcess
Close $kernelProjection 2560.1452746391296 1e-9 'kernel projection'
Close $warmProjection 2672.980880373507 1e-9 'warm projection'
Require ($kernelProjection -gt 900.0) 'kernel projection exceeds frozen gate'
Require ($warmProjection -gt 900.0) 'warm projection exceeds frozen gate'

# Positive direct-core evidence.
Eq ([int]$result.direct_source.curand_init_calls_in_performance_kernel) 0 'performance curand_init count'
Eq ([int]$result.direct_source.curand_normal4_calls_in_performance_kernel) 0 'performance curand_normal4 count'
Eq ([int]$result.direct_source.direct_philox_calls_in_performance_kernel) 1 'performance direct Philox source count'
Eq ([int]$result.direct_source.box_muller_pair_calls_in_performance_kernel) 2 'performance Box-Muller source count'
Require ($source.Contains('const unsigned long long seed64 = base_seed + seed_deltas[bundle];')) 'u64 effective seed in direct core'
Require ($source.Contains('const unsigned long long carry = counter_low < offset_base ? 1ULL : 0ULL;')) 'counter carry in direct core'
Require ($source.Contains('const uint4 raw = curand_Philox4x32_10(counter, key);')) 'direct Philox core call'
Require ($source.Contains('const float2 pair0 = _curand_box_muller(raw.x, raw.y);')) 'first pinned Box-Muller word pair'
Require ($source.Contains('const float2 pair1 = _curand_box_muller(raw.z, raw.w);')) 'second pinned Box-Muller word pair'

# Frozen-gate failures.  These assertions intentionally require the absences
# that make this audit verdict BLOCK rather than silently accepting them.
foreach ($name in @('driver','nvrtc','compiler','compile_options','ptx_sha256','cubin_sha256','compiled_kernel_sha256','torch')) {
    Require (-not ($result.PSObject.Properties.Name -contains $name)) "result top level omits $name binding"
}
Require (-not $resultText.Contains('driver_version')) 'result omits CUDA driver version'
Require (-not $resultText.Contains('nvrtc_version')) 'result omits NVRTC version'
Require (-not $resultText.Contains('compiled_ptx')) 'result omits compiled PTX receipt'
Require (-not $resultText.Contains('compiled_cubin')) 'result omits compiled cubin receipt'
Require (-not $resultText.Contains('compiler_binary_sha256')) 'result omits compiler binary/library hash'
Require (-not $resultText.Contains('compile_options')) 'result omits explicit compile-options receipt'
Require (-not $source.Contains('driverGetVersion')) 'source does not receipt CUDA driver version'
Require (-not $source.Contains('nvrtcVersion')) 'source does not receipt NVRTC version'
Require (-not $source.Contains('import torch')) 'source has no Torch parity path'
Require ($source.Contains('curand_init(seed64, sequences[row], offsets[row] + 4ULL * normal4_indices[row], &state);')) 'shifted-offset one-call reference exists'
Require (-not $source.Contains('curand_init(seed64, sequences[row], offsets[row], &state);')) 'base-offset sequential reference is absent'
Eq ([regex]::Matches($source, 'parity_receipt\s*=\s*run_parity\(\)').Count) 1 'parity gate execution count is one'
Require (-not $source.Contains('for parity_repetition')) 'no three-repetition parity loop'
Require ($result.claim_boundary.Contains('no exact frozen bundle-plan')) 'result discloses missing final plan binding'
Require ($result.claim_boundary.Contains('journal write')) 'result discloses missing journal path'
Require ($result.aggregate.projection_warning.Contains('excludes frozen bundle-plan binding')) 'projection warning binds omitted plan'
Require ($result.aggregate.projection_warning.Contains('journal/global merge')) 'projection warning binds omitted journal/global merge'

Write-Output ("PASS: {0} independent source-only assertions; verdict BLOCK under frozen v1 release prerequisites." -f $script:Checks)
