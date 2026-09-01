$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$result = Get-Content -Raw -LiteralPath (Join-Path $root 'result.json') | ConvertFrom-Json

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Close-Enough([double]$Actual, [double]$Expected) {
    $scale = [Math]::Max(1.0, [Math]::Abs($Expected))
    return [Math]::Abs($Actual - $Expected) -le 1e-12 * $scale
}

$scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root 'autotune.py')).Hash.ToLowerInvariant()
Require ($result.schema -eq 'fuseed_u32_source_free_direct_numeric_autotune_v0') 'schema mismatch'
Require ($result.script_sha256 -eq $scriptHash) 'script hash mismatch'
Require ($scriptHash -eq '22a39ca44b8a0f2c0eaa6f1687db1b7eb9443ab3d0214c2010091205348b3fc1') 'unexpected frozen script'
Require ($result.direct_script.sha256 -eq 'f5a7c8b9a525e02d469ca974f9a6607030b2ca2822b66d4bce31604251516ed5') 'direct-source binding mismatch'
Require ($result.shape.candidates -eq 1048576 -and $result.shape.domains -eq 33) 'shape mismatch'
Require ($result.shape.repetitions -eq 5 -and @($result.rows).Count -eq 5) 'timing matrix mismatch'
Require ($result.derivation.per_domain_xw_only_is_fp32 -eq $true) 'FP32 derivation flag missing'
Require ($result.derivation.common_moments_affine_half_reload_and_q_remain_fp64 -eq $true) 'FP64 remainder flag missing'
Require ($result.parity.raw_bitwise_equal -eq $true -and $result.parity.scaled_bf16_bitwise_equal -eq $true) 'generator parity failed'
Require (@($result.rows.q_sentinel_sha256 | Select-Object -Unique).Count -eq 1) 'launch q sentinels differ'

$winner = @($result.rows | Sort-Object median_seconds,name)[0]
Require ($winner.name -eq $result.winner.name) 'winner selection mismatch'
Require ($winner.name -eq 'warp8_block256_r80') 'unexpected frozen winner'
$projection = [double]$winner.median_seconds * 4096.0 * 3.0
Require (Close-Enough $projection ([double]$winner.projected_three_abi_full_u32_seconds)) 'winner projection mismatch'
Require (Close-Enough $projection 914.5142841339111) 'unexpected frozen projection'
Require ($result.decision.promotion_margin_gate_seconds -eq 800.0) 'gate mismatch'
Require ($projection -gt 800.0 -and $result.decision.winner_projection_below_margin_gate -eq $false) 'runtime kill mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0 -and $result.access.payload_files_opened -eq 0 -and $result.access.network_operations -eq 0) 'access attestation mismatch'

Write-Output 'PASS: direct FP32 autotune is internally consistent and is frozen as an 800-second early kill without Qwen access.'
