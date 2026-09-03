param(
    [string]$Producer = (Join-Path $PSScriptRoot '..\strata_rm_global_swap_v4_wasm_authority')
)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '62bf04cd413317e2e8b98635713419c84394db7b7d2bd4567afddf56957a5e2f'
$ExpectedRoot = 'f535699c4828a02e5769b916b1207309768f7381db5f92a0fb58e10915ae8a25'
$ExpectedV3Manifest = '9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83'
$ExpectedV3Root = '83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad'
$ExpectedV3ReviewManifest = 'ebe65fcf1abd73263be0176cdb70244ebca4f0a883eb6815c24c8956b0d0d89c'
$ExpectedV3ReviewRoot = '3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0'

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "independent static review failed: $Message" }
}

function Hash-Bytes([byte[]]$Bytes) {
    return [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}

$Root = (Resolve-Path -LiteralPath $Producer).Path
$ManifestPath = Join-Path $Root 'source_manifest.json'
$ManifestBytes = [IO.File]::ReadAllBytes($ManifestPath)
Require ((Hash-Bytes $ManifestBytes) -eq $ExpectedManifest) 'producer manifest pin'
$Manifest = [Text.Encoding]::UTF8.GetString($ManifestBytes) | ConvertFrom-Json
Require ($Manifest.source_root_sha256 -eq $ExpectedRoot) 'declared producer root'
Require ($Manifest.v3_manifest_sha256 -eq $ExpectedV3Manifest) 'v3 manifest lineage'
Require ($Manifest.v3_source_root_sha256 -eq $ExpectedV3Root) 'v3 root lineage'
Require ($Manifest.v3_review_manifest_sha256 -eq $ExpectedV3ReviewManifest) 'v3 review manifest lineage'
Require ($Manifest.v3_review_source_root_sha256 -eq $ExpectedV3ReviewRoot) 'v3 review root lineage'

$Observed = @()
foreach ($Row in $Manifest.members) {
    $Path = Join-Path $Root $Row.name
    $Bytes = [IO.File]::ReadAllBytes($Path)
    $Digest = Hash-Bytes $Bytes
    Require ($Bytes.Length -eq $Row.bytes) "member bytes $($Row.name)"
    Require ($Digest -eq $Row.sha256) "member hash $($Row.name)"
    $Observed += [ordered]@{bytes=[int64]$Bytes.Length;name=[string]$Row.name;sha256=$Digest}
}
$Canonical = ConvertTo-Json -InputObject $Observed -Compress -Depth 5
$ComputedRoot = Hash-Bytes ([Text.Encoding]::ASCII.GetBytes($Canonical))
Require ($ComputedRoot -eq $ExpectedRoot) 'recomputed canonical row root'
$Entries = @(Get-ChildItem -Force -LiteralPath $Root)
Require ($Entries.Count -eq 10) 'exact package entry count'
Require (@($Entries | Where-Object { $_.PSIsContainer }).Count -eq 0) 'flat regular closure'
Require ($Manifest.members.Count -eq 9) 'nine manifest members'

$Authority = Get-Content -Raw -LiteralPath (Join-Path $Root 'authority_v4.py')
$Sandbox = Get-Content -Raw -LiteralPath (Join-Path $Root 'wasm_runtime_sandbox.py')
$Gate = Get-Content -Raw -LiteralPath (Join-Path $Root 'run_source_gate.py')
$Status = Get-Content -Raw -LiteralPath (Join-Path $Root 'EXECUTION_STATUS.json') | ConvertFrom-Json

Require ($Authority.Contains('authenticate_runtime_audit_package')) 'runtime audit authenticator'
Require ($Authority.Contains('authenticate_semantic_decoder_audit_package')) 'semantic audit authenticator'
Require ($Authority.Contains('expected_runtime_tree_root_sha256')) 'runtime tree external pin'
Require ($Authority.Contains('native_libraries_loaded_and_rehashed')) 'native audit receipt requirement'

$Consume = $Sandbox.IndexOf('config.consume_fuel = True')
$Compile = $Sandbox.IndexOf('decoder_module = wasmtime.Module')
$DecoderStore = $Sandbox.IndexOf('decoder_store = configure_store')
$DecoderInstantiate = $Sandbox.IndexOf('decoder_instance = linker.instantiate')
$EncoderStore = $Sandbox.IndexOf('encoder_store = configure_store')
$EncoderInstantiate = $Sandbox.IndexOf('encoder_instance = wasmtime.Instance')
Require ($Consume -ge 0 -and $Consume -lt $Compile) 'fuel-enabled engine before compilation'
Require ($Sandbox.Contains('store.set_limits(memory_size=STORE_MEMORY_LIMIT_BYTES')) 'Store limiter call'
Require ($Sandbox.Contains('store.set_fuel(fuel)')) 'Store fuel call'
Require ($DecoderStore -ge 0 -and $DecoderStore -lt $DecoderInstantiate) 'decoder Store before instantiation'
Require ($EncoderStore -ge 0 -and $EncoderStore -lt $EncoderInstantiate) 'encoder Store before instantiation'

Require ($Sandbox.Contains('packet_payload = regular_bytes(args.packet')) 'host packet bytes snapshot'
Require ($Sandbox.Contains('def read_packet(caller, offset, destination, length)')) 'bounded callback'
Require ($Sandbox.Contains('for left, right in read_intervals')) 'overlap rejection'
Require ($Sandbox.Contains('memory.write(caller, packet_payload[offset:end], destination)')) 'slice-only packet copy'
Require ($Sandbox.Contains('decoder receives exactly one bounded packet callback')) 'sole decoder import contract'
Require ($Sandbox.Contains('encoder_import_list == []')) 'zero-import canonical encoder'
Require ($Sandbox.Contains('wasmtime.Instance(encoder_store, encoder_module, [])')) 'independent encoder instance'
Require (-not $Sandbox.Contains('linker.define(encoder_store')) 'no encoder packet callback binding'

$SemanticRead = $Sandbox.IndexOf('semantic_state = bytes(decoder_memory.read')
$SemanticWrite = $Sandbox.IndexOf('encoder_memory.write(encoder_store, semantic_state')
Require ($SemanticRead -ge 0 -and $SemanticRead -lt $SemanticWrite) 'semantic-only encoder transfer'
$SemanticBridge = $Sandbox.Substring($SemanticRead, $SemanticWrite - $SemanticRead)
Require (-not $SemanticBridge.Contains('strict_json(semantic_state')) 'semantic state remains opaque'

Require ($Sandbox.Contains('expected <= observed')) 'all bundled native images must be mapped'
Require ($Sandbox.Contains('loaded_from_snapshot == expected')) 'snapshot native closure'
Require (-not $Sandbox.Contains('sys.implementation.cache_tag')) 'current Python ABI not reobserved'
Require (-not $Sandbox.Contains('platform.machine')) 'current machine target not reobserved'

Require ($Authority.Contains('v3.evaluate_acceptance(results, scientific["record"], enforce=True)')) 'pinned v3 acceptance reuse'
Require ($Authority.Contains('one packet for every audited route')) 'one packet per route'
Require ($Authority.Contains('distinct expert packet per route')) 'packet aliases rejected'
Require ($Gate.Contains('runtime_audit_package_opened') -and
         $Gate.Contains('model_data_accessed') -and
         $Gate.Contains('payloads_opened')) 'source gate records held inputs'
Require ($Status.wasmtime_runtime_imported -eq $false -and
         $Status.wasm_guest_executed -eq $false -and
         $Status.model_data_accessed -eq $false -and
         $Status.payloads_opened -eq 0) 'runtime and payload status held'

$Result = [ordered]@{
    schema = 'strata-rm-global-swap-v4-independent-static-powershell-receipt'
    producer_manifest_sha256 = $ExpectedManifest
    producer_source_root_sha256 = $ComputedRoot
    producer_members = 9
    exact_flat_closure = $true
    preinstantiation_store_limits_and_fuel = $true
    immutable_host_packet_callback = $true
    distinct_zero_import_canonical_encoder = $true
    semantic_state_runtime_parsed = $false
    current_host_abi_platform_target_reobserved = $false
    native_closure_scope = 'copied_runtime_snapshot_only'
    v3_scientific_family_control_and_read_acceptance_reused = $true
    wasmtime_executed = $false
    payloads_opened = 0
    status = 'PASS_STATIC_REPAIR_MECHANISMS__CONDITIONAL_EXTERNAL_AUDITS__RUNTIME_AND_PAYLOAD_HELD'
}
$Result | ConvertTo-Json -Depth 5
