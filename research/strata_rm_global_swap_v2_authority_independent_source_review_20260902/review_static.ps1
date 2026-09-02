param(
    [Parameter(Mandatory = $true)] [string]$Producer,
    [Parameter(Mandatory = $true)] [string]$Output
)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '1f1caf2884a8b0b8713f213a16a0a32194238b64969e9d9cf3aaa339ddb776be'
$ExpectedRoot = 'e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9'

function Get-LowerSha256([byte[]]$Bytes) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
}

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "static review failed: $Message" }
}

function Json-String([string]$Value) {
    return ($Value | ConvertTo-Json -Compress)
}

$producerPath = (Resolve-Path -LiteralPath $Producer).Path
$producerItem = Get-Item -LiteralPath $producerPath -Force
Require ($producerItem.PSIsContainer) 'producer is a directory'
Require (($producerItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0) 'producer root is not a reparse point'

$manifestPath = Join-Path $producerPath 'source_manifest.json'
$manifestBytes = [System.IO.File]::ReadAllBytes($manifestPath)
$manifestSha = Get-LowerSha256 $manifestBytes
Require ($manifestSha -eq $ExpectedManifest) 'manifest external pin'
$manifest = [System.Text.Json.JsonSerializer]::Deserialize[System.Text.Json.JsonElement]($manifestBytes)
Require ($manifest.GetProperty('schema').GetString() -eq 'strata-rm-global-swap-v2-authority-source-manifest') 'manifest schema'
Require ($manifest.GetProperty('source_root_sha256').GetString() -eq $ExpectedRoot) 'manifest source root pin'

$rows = @()
$expectedNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($member in $manifest.GetProperty('members').EnumerateArray()) {
    $name = $member.GetProperty('name').GetString()
    Require ($expectedNames.Add($name)) "unique member $name"
    Require ([System.IO.Path]::GetFileName($name) -eq $name) "flat member $name"
    $path = Join-Path $producerPath $name
    $item = Get-Item -LiteralPath $path -Force
    Require (-not $item.PSIsContainer) "regular member $name"
    Require (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0) "non-link member $name"
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $hash = Get-LowerSha256 $bytes
    $expectedBytes = $member.GetProperty('bytes').GetInt64()
    $expectedSha = $member.GetProperty('sha256').GetString()
    Require ($bytes.LongLength -eq $expectedBytes) "member byte count $name"
    Require ($hash -eq $expectedSha) "member hash $name"
    $rows += '{"bytes":' + $expectedBytes + ',"name":' + (Json-String $name) + ',"sha256":' + (Json-String $expectedSha) + '}'
}

$actualNames = @(Get-ChildItem -LiteralPath $producerPath -Force | ForEach-Object { $_.Name })
$actualClosure = @($actualNames | Sort-Object) -join "`n"
$expectedClosureRows = @($expectedNames) + 'source_manifest.json'
$expectedClosure = @($expectedClosureRows | Sort-Object) -join "`n"
Require ($actualClosure -ceq $expectedClosure) 'exact flat closure'

$canonicalRows = '[' + ($rows -join ',') + ']'
$rootSha = Get-LowerSha256 ([System.Text.Encoding]::ASCII.GetBytes($canonicalRows))
Require ($rootSha -eq $ExpectedRoot) 'independent canonical member root'

$authority = [System.IO.File]::ReadAllText((Join-Path $producerPath 'authority_v2.py'))
$current = [System.IO.File]::ReadAllText((Join-Path $producerPath 'current_snapshot_worker.py'))
$parity = [System.IO.File]::ReadAllText((Join-Path $producerPath 'parity_worker.py'))
$independent = [System.IO.File]::ReadAllText((Join-Path $producerPath 'independent_rm_order.py'))
$launcher = [System.IO.File]::ReadAllText((Join-Path $producerPath 'instrumented_decoder_worker.py'))

$checks = [ordered]@{
    root_lstat_before_resolve = ($authority.IndexOf('before = original.lstat()') -ge 0 -and $authority.IndexOf('before = original.lstat()') -lt $authority.IndexOf('resolved = original.resolve(strict=True)'))
    external_snapshot_before_worker = $authority.Contains('snapshot_pinned_files(external_root, EXTERNAL_PINS, external_snapshot)')
    decoder_and_launcher_snapshotted = ($authority.Contains('_write_immutable(decoder, decoder_payload') -and $authority.Contains('_write_immutable(launcher, launcher_payload'))
    separate_decoder_audit_capability = ($authority.Contains('def authenticate_decoder_audit_capability(') -and $authority.Contains('PASS_INDEPENDENT_DECODER_AUDIT_V2'))
    launcher_owned_interval_records = $launcher.Contains('self._operations.append({"offset": before, "length": after - before})')
    scientific_capability_separate_from_commitment = ($authority.Contains('def authenticate_scientific_capability(') -and $authority.Contains('commitment uses exact auditor capability case set'))
    per_family_target_loop = ($authority.Contains('for family in scientific["architecture_families"]:') -and $authority.Contains('family {family}: target/control/read acceptance'))
    strongest_control_subtracted = ($authority.Contains('strongest_name, strongest = max(') -and $authority.Contains('advantage >= MIN_SOURCE_SPECIFIC_BPW'))
    both_global_lengths_invoked = ($current.Contains('for n in TARGET_N:') -and $independent.Contains('TARGET_N = (1 << 20, 1 << 21)'))
    independent_cpu_gosper = ($independent.Contains('low = value & -value') -and $independent.Contains('for weight in range(width, -1, -1):'))
    independent_gpu_byte_lut = ($independent.Contains('table = cp.asarray([int(value).bit_count() for value in range(256)]') -and $independent.Contains('return cp.argsort(key)'))
    parity_compares_cpu_gpu_and_v1 = ($parity.Contains('np.array_equal(cpu, producer_cpu)') -and $parity.Contains('np.array_equal(cpu, gpu_host)'))
}
foreach ($entry in $checks.GetEnumerator()) {
    Require ([bool]$entry.Value) "source invariant $($entry.Key)"
}

$receipt = [ordered]@{
    schema = 'strata-rm-global-swap-v2-authority-independent-static-review-receipt'
    producer_manifest_sha256 = $manifestSha
    producer_source_root_sha256 = $rootSha
    producer_members = $rows.Count
    exact_flat_closure = $true
    independent_canonical_root = $true
    source_invariants = $checks
    model_data_accessed = $false
    payloads_opened = 0
    python_tests_executed = $false
    cupy_workers_executed = $false
    disposition = 'PASS_STATIC_CLOSURE_AND_SUBSTANTIAL_V1_REPAIRS__BLOCK_PHYSICAL_AUTHORITY_ON_SCIENTIFIC_PROVENANCE_AND_ROUTED_EXPERT_IO__HOLD_PAYLOAD_AND_RD'
}
$json = $receipt | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($Output, $json + "`n", [System.Text.UTF8Encoding]::new($false))
$json
