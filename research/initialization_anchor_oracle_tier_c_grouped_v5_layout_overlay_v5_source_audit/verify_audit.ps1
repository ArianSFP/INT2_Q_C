[CmdletBinding()]
param(
    [string]$Producer = (Join-Path $PSScriptRoot '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5'),
    [switch]$Replay
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'AUDIT VERIFY FAILED: pwsh 7 or newer is required' }
$here = $PSScriptRoot

function Sha256-File([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}
function Sha256-Bytes([byte[]]$Bytes) {
    return [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($Bytes)
    ).ToLowerInvariant()
}
function Sort-JsonNode([object]$Value) {
    if ($Value -is [System.Collections.IDictionary]) {
        $ordered = [ordered]@{}
        foreach ($key in ($Value.Keys | Sort-Object {[string]$_} -CaseSensitive)) {
            $ordered[[string]$key] = Sort-JsonNode $Value[$key]
        }
        return $ordered
    }
    if (($Value -is [System.Collections.IEnumerable]) -and -not ($Value -is [string])) {
        $items = [System.Collections.Generic.List[object]]::new()
        foreach ($item in $Value) { $items.Add((Sort-JsonNode $item)) }
        return $items.ToArray()
    }
    return $Value
}
function Internal-Sha256([System.Collections.IDictionary]$Value, [string]$Field) {
    $copy = [ordered]@{}
    foreach ($key in $Value.Keys) {
        if ([string]$key -ne $Field) { $copy[[string]$key] = $Value[$key] }
    }
    $options = [System.Text.Json.JsonSerializerOptions]::new()
    $options.Encoder = [System.Text.Encodings.Web.JavaScriptEncoder]::UnsafeRelaxedJsonEscaping
    $json = [System.Text.Json.JsonSerializer]::Serialize((Sort-JsonNode $copy), $options)
    return Sha256-Bytes ([Text.Encoding]::UTF8.GetBytes($json))
}
function Require-ExactKeys(
    [System.Collections.IDictionary]$Value,
    [string[]]$Expected,
    [string]$Label
) {
    $observed = @($Value.Keys | ForEach-Object {[string]$_})
    if ($observed.Count -ne $Expected.Count) { throw "$Label key count mismatch" }
    foreach ($key in $Expected) {
        if (-not $Value.Contains($key)) { throw "$Label missing key: $key" }
    }
}
function Require-Zero([System.Collections.IDictionary]$Value, [string[]]$Fields, [string]$Label) {
    foreach ($field in $Fields) {
        if ([int64]$Value[$field] -ne 0) { throw "$Label must be zero: $field" }
    }
}
function Require-Close([double]$Actual, [double]$Expected, [double]$Tolerance, [string]$Label) {
    if ([Math]::Abs($Actual - $Expected) -gt $Tolerance) { throw "$Label mismatch" }
}

# Authenticate this audit as one exact flat, regular, non-reparse closure.
$allowed = @(
    'ARTIFACT_SHA256SUMS.txt',
    'README.md',
    'audit_receipt.json',
    'run_source_audit.ps1',
    'verify_audit.ps1'
)
$members = @(Get-ChildItem -LiteralPath $here -Force)
if ($members.Count -ne $allowed.Count) { throw 'audit package member count mismatch' }
foreach ($member in $members) {
    if ($member.Name -cnotin $allowed) { throw "unexpected audit package member: $($member.Name)" }
    if ($member.PSIsContainer -or
        (($member.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "audit member is not a regular non-reparse file: $($member.Name)"
    }
}

$manifestPath = Join-Path $here 'ARTIFACT_SHA256SUMS.txt'
$manifestBytes = [IO.File]::ReadAllBytes($manifestPath)
$manifestText = [Text.Encoding]::ASCII.GetString($manifestBytes)
if ($manifestText.Contains("`r", [StringComparison]::Ordinal) -or
    -not $manifestText.EndsWith("`n", [StringComparison]::Ordinal)) {
    throw 'audit manifest must be ASCII LF terminated'
}
$manifestEntries = [ordered]@{}
$previous = $null
foreach ($line in $manifestText.Split("`n", [StringSplitOptions]::RemoveEmptyEntries)) {
    if ($line -cnotmatch '^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_.-]*)$') {
        throw 'audit artifact manifest grammar mismatch'
    }
    $name = $Matches[2]
    if ($manifestEntries.Contains($name)) { throw "duplicate audit artifact: $name" }
    if ($null -ne $previous -and [StringComparer]::Ordinal.Compare($previous, $name) -ge 0) {
        throw 'audit artifact manifest is not strictly sorted'
    }
    $manifestEntries[$name] = $Matches[1]
    $previous = $name
}
$manifested = @('README.md','audit_receipt.json','run_source_audit.ps1','verify_audit.ps1')
if ($manifestEntries.Count -ne $manifested.Count) { throw 'audit manifest count mismatch' }
foreach ($name in $manifested) {
    if (-not $manifestEntries.Contains($name)) { throw "missing audit artifact: $name" }
    if ((Sha256-File (Join-Path $here $name)) -ne $manifestEntries[$name]) {
        throw "audit artifact hash mismatch: $name"
    }
}

# Authenticate receipt semantics and its canonical internal seal.
$receiptRaw = [IO.File]::ReadAllText(
    (Join-Path $here 'audit_receipt.json'),
    [Text.UTF8Encoding]::new($false, $true)
)
$receipt = $receiptRaw | ConvertFrom-Json -AsHashtable -Depth 100
Require-ExactKeys $receipt @(
    'schema','status','audited_utc','audited_target','authentication',
    'independent_source_only_execution','verified_v4_blocker_repairs',
    'scientific_and_ledger_replay','adversarial_tests','release_blockers',
    'access_ledger','authorization','verdict','audit_receipt_sha256'
) 'audit receipt'
if ((Internal-Sha256 $receipt 'audit_receipt_sha256') -ne $receipt['audit_receipt_sha256']) {
    throw 'audit receipt internal seal mismatch'
}
if ($receipt['schema'] -ne 'tier_c_grouped_v5_layout_overlay_v5_independent_source_audit_block_receipt_v1') {
    throw 'audit receipt schema mismatch'
}
if ($receipt['status'] -ne 'BLOCKED_AUTHENTICATED_DISPATCH_AND_ATOMIC_TOCTOU_REPAIRS_REQUIRED') {
    throw 'audit receipt status mismatch'
}
if ($receipt['verdict'] -ne 'BLOCK_DO_NOT_AUTHORIZE_CALIBRATION_OR_PRODUCTION') {
    throw 'audit receipt verdict mismatch'
}

# Rebind the named audited producer identity without running producer code.
$producerPath = [IO.Path]::GetFullPath($Producer)
$target = $receipt['audited_target']
$targetHashes = [ordered]@{
    artifact_manifest_sha256 = Sha256-File (Join-Path $producerPath 'ARTIFACT_SHA256SUMS.txt')
    candidate_lock_file_sha256 = Sha256-File (Join-Path $producerPath 'candidate_lock.json')
    runner_sha256 = Sha256-File (Join-Path $producerPath 'tier_c_gate.py')
    common_sha256 = Sha256-File (Join-Path $producerPath 'common.py')
    bootstrap_sha256 = Sha256-File (Join-Path $producerPath 'verify_prelaunch.py')
    kernels_sha256 = Sha256-File (Join-Path $producerPath 'kernels.py')
    overlay_sha256 = Sha256-File (Join-Path $producerPath 'overlay.py')
    parity_sha256 = Sha256-File (Join-Path $producerPath 'parity.py')
    source_trace_sha256 = Sha256-File (Join-Path $producerPath 'source_trace.py')
}
foreach ($key in $targetHashes.Keys) {
    if ($target[$key] -ne $targetHashes[$key]) { throw "producer target identity mismatch: $key" }
}
if ($target['artifact_count'] -ne 16 -or $target['exact_package_member_count'] -ne 17) {
    throw 'producer target closure accounting mismatch'
}

$lockRaw = [IO.File]::ReadAllText(
    (Join-Path $producerPath 'candidate_lock.json'),
    [Text.UTF8Encoding]::new($false, $true)
)
$sealMatches = [regex]::Matches($lockRaw, '("lock_sha256"\s*:\s*")([0-9a-f]{64})(")')
if ($sealMatches.Count -ne 1) { throw 'candidate lock internal seal cardinality mismatch' }
$seal = $sealMatches[0]
$normalized = (
    $lockRaw.Substring(0, $seal.Groups[2].Index) +
    'TO_BE_FILLED_AFTER_CANONICAL_FREEZE' +
    $lockRaw.Substring($seal.Groups[2].Index + $seal.Groups[2].Length)
)
if ((Sha256-Bytes ([Text.Encoding]::UTF8.GetBytes($normalized))) -ne
    $target['candidate_lock_internal_sha256']) {
    throw 'candidate lock internal seal mismatch'
}

# Pin the BLOCK findings, replay identities, and zero-access/authorization ledger.
if ($receipt['release_blockers'].Count -ne 3) { throw 'release blocker count mismatch' }
$blockerIds = @(
    'AUTHENTICATED_DISPATCH_CAPABILITY_ABSENT',
    'BOUNDARY_REVALIDATE_AND_CREATE_NOT_ATOMIC',
    'AUTHENTICATED_PACKAGE_BYTES_REOPENED_FOR_IMPORT'
)
for ($index = 0; $index -lt $blockerIds.Count; $index++) {
    if ($receipt['release_blockers'][$index]['id'] -ne $blockerIds[$index]) {
        throw "release blocker identity mismatch at index $index"
    }
}
$repairs = $receipt['verified_v4_blocker_repairs']
if ($repairs['raw_entrypoint_component_object_and_mount_binding']['status'] -ne 'PASS_AT_CHECK_INSTANTS' -or
    $repairs['literal_direct_script_rejection']['status'] -ne 'PASS' -or
    $repairs['pairwise_path_object_and_mount_disjointness']['status'] -ne 'PASS_AT_CHECK_INSTANTS' -or
    $repairs['closure_before_output_or_journal']['status'] -ne 'PASS') {
    throw 'v4 repair finding mismatch'
}
$execution = $receipt['independent_source_only_execution']
if ($execution['assertions_passed'] -ne 209 -or $execution['assertions_failed'] -ne 0) {
    throw 'independent execution assertion accounting mismatch'
}
if ($execution['adversarial_evidence_sha256'] -ne
    '1b6e73051b53e40fec660d28baa672d1711b538e3c5665dfa69556a3b9fc2ce3') {
    throw 'adversarial evidence identity mismatch'
}
$ledger = $receipt['scientific_and_ledger_replay']
if ($ledger['status'] -ne 'PASS_WITHIN_SOURCE_SCOPE' -or
    [int64]$ledger['stage0_generated_values'] -ne 21609054208 -or
    [int64]$ledger['stage1_generated_values'] -ne 6572408832 -or
    [int64]$ledger['post_selection_reporting_generated_values'] -ne 2162688 -or
    [int64]$ledger['end_to_end_generated_values'] -ne 28183625728) {
    throw 'scientific arithmetic replay mismatch'
}
Require-Close ([double]$ledger['side_bpw']) (80.0 * 8.0 / 28311552.0) 1e-18 'side bpw'
Require-Close ([double]$ledger['maximum_compatible_base_codec_bpw']) (2.15 - 80.0 * 8.0 / 28311552.0) 1e-15 'base codec cap'
Require-Close ([double]$ledger['conservative_appended_cold_read_amplification']) (1.169444 + 20.0 / (4718592.0 * 2.15 / 8.0)) 1e-15 'read amplification'

Require-Zero $receipt['access_ledger'] @(
    'payload_paths_supplied','payload_files_opened','payload_manifest_or_directory_operations',
    'python_runtime_imports','numeric_runtime_imports','accelerator_initializations',
    'accelerator_jobs','network_operations','production_runs','producer_files_modified'
) 'access ledger'
foreach ($field in @(
    'source_package_passed','source_free_calibration_authorized',
    'payload_or_manifest_launch_authorized','production_run_authorized',
    'implementation_authorized_by_this_audit'
)) {
    if ($receipt['authorization'][$field] -ne $false) { throw "authorization must be false: $field" }
}

if ($Replay) {
    $raw = (& (Join-Path $here 'run_source_audit.ps1') -Producer $producerPath | Out-String)
    $replayed = $raw | ConvertFrom-Json -AsHashtable -Depth 100
    if ($replayed['status'] -ne $receipt['status']) { throw 'replay status mismatch' }
    if ($replayed['assertions_passed'] -ne $execution['assertions_passed']) {
        throw 'replay assertion accounting mismatch'
    }
    if ($replayed['adversarial_evidence_sha256'] -ne $execution['adversarial_evidence_sha256']) {
        throw 'replay evidence hash mismatch'
    }
    if ($replayed['producer_manifest_sha256'] -ne $target['artifact_manifest_sha256'] -or
        $replayed['candidate_lock_file_sha256'] -ne $target['candidate_lock_file_sha256'] -or
        $replayed['candidate_lock_internal_sha256'] -ne $target['candidate_lock_internal_sha256']) {
        throw 'replay target identity mismatch'
    }
    if ($replayed['verdict'] -ne $receipt['verdict']) { throw 'replay verdict mismatch' }
}

Write-Output 'grouped-v5 layout-overlay v5 independent source audit verified: release BLOCKED'
