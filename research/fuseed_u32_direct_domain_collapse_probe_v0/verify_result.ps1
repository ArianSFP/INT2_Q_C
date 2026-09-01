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

$scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root 'probe.py')).Hash.ToLowerInvariant()
Require ($result.schema -eq 'fuseed_u32_source_free_direct_domain_collapse_probe_v0') 'schema mismatch'
Require ($result.script_sha256 -eq $scriptHash) 'script hash mismatch'
Require ($scriptHash -eq 'd3274d82e5321a33f0850ca02251635d41fefc804dd3d5960bac1c01cbab971a') 'unexpected frozen script'
Require ($result.direct_script.sha256 -eq 'f5a7c8b9a525e02d469ca974f9a6607030b2ca2822b66d4bce31604251516ed5') 'direct-source binding mismatch'
Require ($result.shape.candidates -eq 1048576 -and $result.shape.frozen_full_domains -eq 33) 'shape mismatch'
Require ($result.shape.repetitions -eq 5 -and @($result.rows).Count -eq 24) 'timing matrix mismatch'
Require (($result.derivation.active_domain_counts -join ',') -eq '1,2,4,8,16,33') 'domain grid mismatch'
Require ($result.derivation.cross_moments_affine_half_reload_and_q_remain_fp64 -eq $true) 'exact FP64 flag missing'
Require ($result.derivation.generator_counter_box_muller_scale_and_bf16_path_unchanged -eq $true) 'generator preservation flag missing'
Require ($result.parity.raw_bitwise_equal -eq $true -and $result.parity.scaled_bf16_bitwise_equal -eq $true) 'generator parity failed'

$observedByDomain = @{}
foreach ($row in $result.rows) {
    for ($domain = 0; $domain -lt [int]$row.active_domains; $domain++) {
        if (-not $observedByDomain.ContainsKey($domain)) { $observedByDomain[$domain] = @{} }
        $observedByDomain[$domain][[string]$row.active_domain_q_sentinel_sha256[$domain]] = $true
    }
}
for ($domain = 0; $domain -lt 33; $domain++) {
    Require ($observedByDomain.ContainsKey($domain) -and $observedByDomain[$domain].Count -eq 1) "domain sentinel mismatch: $domain"
}

foreach ($count in @(1,2,4,8,16,33)) {
    $winner = @($result.rows | Where-Object {[int]$_.active_domains -eq $count} | Sort-Object median_seconds,launch_name)[0]
    $published = $result.winners_by_active_domain_count."$count"
    Require ($winner.launch_name -eq $published.launch_name) "winner mismatch: $count"
    Require (Close-Enough ([double]$winner.median_seconds) ([double]$published.median_seconds)) "winner time mismatch: $count"
    $projection = [double]$winner.median_seconds * 4096.0 * 3.0
    Require (Close-Enough $projection ([double]$winner.projected_three_abi_full_u32_kernel_seconds)) "projection mismatch: $count"
}

$sourceWinner = $result.winners_by_active_domain_count.'1'
$threeAbi = [double]$sourceWinner.projected_three_abi_full_u32_kernel_seconds
Require ($sourceWinner.launch_name -eq 'block256') 'unexpected source-only launch winner'
Require (Close-Enough $threeAbi 1537.884216785431) 'unexpected frozen three-ABI projection'
Require (Close-Enough ($threeAbi / 3.0) 512.6280722618103) 'one-ABI inference mismatch'
Require ($result.decision.prospective_source_only_kernel_margin_gate_seconds -eq 800.0) 'gate mismatch'
Require ($threeAbi -gt 800.0 -and $result.decision.source_only_kernel_projection_below_margin_gate -eq $false) 'three-ABI runtime kill mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0 -and $result.access.payload_files_opened -eq 0 -and $result.access.network_operations -eq 0) 'access attestation mismatch'

Write-Output 'PASS: exact domain-collapse probe is internally consistent, kills three ABIs, and only motivates a distinct one-ABI design.'
