param(
    [Parameter(Mandatory = $true)] [string]$Producer,
    [Parameter(Mandatory = $true)] [string]$Output
)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83'
$ExpectedRoot = '83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad'

function Get-LowerSha256([byte[]]$Bytes) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
}

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "v3 static review failed: $Message" }
}

function Json-String([string]$Value) {
    return ($Value | ConvertTo-Json -Compress)
}

$root = (Resolve-Path -LiteralPath $Producer).Path
$rootItem = Get-Item -LiteralPath $root -Force
Require ($rootItem.PSIsContainer) 'producer directory'
Require (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'real producer root'
$manifestBytes = [IO.File]::ReadAllBytes((Join-Path $root 'source_manifest.json'))
$manifestHash = Get-LowerSha256 $manifestBytes
Require ($manifestHash -eq $ExpectedManifest) 'manifest external pin'
$manifest = [Text.Json.JsonSerializer]::Deserialize[Text.Json.JsonElement]($manifestBytes)
Require ($manifest.GetProperty('schema').GetString() -eq 'strata-rm-global-swap-v3-physical-authority-source-manifest') 'manifest schema'
Require ($manifest.GetProperty('source_root_sha256').GetString() -eq $ExpectedRoot) 'manifest root pin'

$names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$canonical = @()
foreach ($row in $manifest.GetProperty('members').EnumerateArray()) {
    $name = $row.GetProperty('name').GetString()
    Require ($names.Add($name)) "unique member $name"
    Require ([IO.Path]::GetFileName($name) -eq $name) "flat member $name"
    $item = Get-Item -LiteralPath (Join-Path $root $name) -Force
    Require (-not $item.PSIsContainer) "regular member $name"
    Require (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "non-link member $name"
    $bytes = [IO.File]::ReadAllBytes($item.FullName)
    $length = $row.GetProperty('bytes').GetInt64()
    $hash = $row.GetProperty('sha256').GetString()
    Require ($bytes.LongLength -eq $length) "length $name"
    Require ((Get-LowerSha256 $bytes) -eq $hash) "hash $name"
    $canonical += '{"bytes":' + $length + ',"name":' + (Json-String $name) + ',"sha256":' + (Json-String $hash) + '}'
}
$actual = @((Get-ChildItem -LiteralPath $root -Force).Name | Sort-Object) -join "`n"
$wantedRows = @($names) + 'source_manifest.json'
$wanted = @($wantedRows | Sort-Object) -join "`n"
Require ($actual -ceq $wanted) 'exact flat closure'
$rootHash = Get-LowerSha256 ([Text.Encoding]::ASCII.GetBytes('[' + ($canonical -join ',') + ']'))
Require ($rootHash -eq $ExpectedRoot) 'independent canonical member root'

$authority = [IO.File]::ReadAllText((Join-Path $root 'authority_v3.py'))
$sandbox = [IO.File]::ReadAllText((Join-Path $root 'wasm_decoder_sandbox.py'))
$checks = [ordered]@{
    scientific_manifest_root_receipt_capability_pins = ($authority.Contains('def authenticate_scientific_audit_package(') -and $authority.Contains('expected_receipt_sha256') -and $authority.Contains('expected_capability_sha256'))
    scientific_exact_closure = ($authority.Contains('scientific audit exact closure') -and $authority.Contains('PASS_INDEPENDENT_SCIENTIFIC_PROVENANCE_AUDIT_V3'))
    cross_family_checkpoint_tensor_identity_schema_rejected = ($authority.Contains('for field in ("checkpoint_manifest_sha256", "tensor_manifest_sha256",') -and $authority.Contains('cross-family {field} alias'))
    source_path_and_hash_aliases_rejected = ($authority.Contains('cross-family source-byte alias') -and $authority.Contains('source-path alias across routed cases'))
    one_gate_up_down_triplet = ($authority.Contains('len(rows) == 3') -and $authority.Contains('one compatible routed expert'))
    distinct_packet_per_route = ($authority.Contains('one distinct expert-local packet per route') -and $authority.Contains('one packet for every exact audited route'))
    exact_page_ledger = ($authority.Contains('page_count = (len(packet) + PAGE_BYTES - 1) // PAGE_BYTES') -and $authority.Contains('cold_read_amplification'))
    maximum_per_family_read_enforced = ($authority.Contains('maximum_routed_expert_cold_read_amplification') -and $authority.Contains('MAX_COLD_READ_AMPLIFICATION'))
    zero_wasm_imports = ($sandbox.Contains('module_imports == []') -and $sandbox.Contains('wasmtime.Instance(store, module, [])'))
    no_wasi_linker = (-not $sandbox.Contains('Linker(') -and -not $sandbox.Contains('define_wasi'))
    disjoint_packet_reconstruction_canonical_regions = ($sandbox.Contains('reconstruction_offset = align64(len(supplied_packet))') -and $sandbox.Contains('canonical_offset = align64(reconstruction_offset + reconstruction_bytes)'))
    padded_packet_final_state_checked = ($sandbox.Contains('post_decode_packet == supplied_packet'))
    memory_bound_checked = ($sandbox.Contains('MAX_LINEAR_MEMORY_BYTES') -and $sandbox.Contains('decoder may not grow memory beyond bound'))
    canonical_bytes_checked = ($sandbox.Contains('canonical == packet_payload') -and $authority.Contains('canonical == packet'))
}
foreach ($entry in $checks.GetEnumerator()) { Require ([bool]$entry.Value) "source invariant $($entry.Key)" }

$limitations = [ordered]@{
    transient_packet_writes_not_observed = ($sandbox.IndexOf('post_decode_packet = bytes(memory.read(') -gt $sandbox.IndexOf('canonical_reencode('))
    store_memory_limiter_absent = (-not $sandbox.Contains('set_limits') -and -not $sandbox.Contains('StoreLimits'))
    fuel_or_epoch_budget_absent = (-not $sandbox.Contains('set_fuel') -and -not $sandbox.Contains('epoch_deadline'))
    wasmtime_version_and_binary_hash_unreported = (-not $sandbox.Contains('__version__') -and -not $sandbox.Contains('wasmtime_sha256'))
    runtime_canonicality_is_byte_replay_only = $sandbox.Contains('canonical == packet_payload')
}

$receipt = [ordered]@{
    schema = 'strata-rm-global-swap-v3-physical-authority-independent-static-review-receipt'
    producer_manifest_sha256 = $manifestHash
    producer_source_root_sha256 = $rootHash
    producer_members = $canonical.Count
    exact_flat_closure = $true
    source_mechanism_checks = $checks
    residual_limit_observations = $limitations
    model_data_accessed = $false
    payloads_opened = 0
    python_tests_executed = $false
    wasmtime_executed = $false
    disposition = 'PASS_V3_STATIC_AUTHORITY_REPAIRS__CONDITIONAL_ON_PINNED_WASMTIME_AND_DECODER_AUDIT_OF_TRANSIENT_MUTATION_MEMORY_AND_CANONICALITY__HOLD_PAYLOAD_AND_RD'
}
$json = $receipt | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($Output, $json + "`n", [Text.UTF8Encoding]::new($false))
$json
