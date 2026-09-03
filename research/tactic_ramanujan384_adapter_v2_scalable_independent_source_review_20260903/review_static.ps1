param(
    [Parameter(Mandatory = $true)] [string]$Producer,
    [Parameter(Mandatory = $true)] [string]$ExternalVerifier,
    [Parameter(Mandatory = $true)] [string]$Output
)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209'
$ExpectedRoot = 'bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495'
$ExpectedVerifier = '74f5a56f1371f67ffa4e83ea34b761c2de61ea0900e3374cc25092f2d333e92c'
$Disposition = 'PASS_V2_STATIC_SCALABILITY_AND_SOURCE_FREE_CONTROL_MECHANISM__HOLD_CUPY_QWEN_AND_PRODUCTION_COARSE_DECODER_UNTIL_ATOMIC_CLOSURE_AND_HARDENED_EXECUTION'

function Get-LowerSha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "Ramanujan v2 static review failed: $Message" }
}
function Json-String([string]$Value) { return ($Value | ConvertTo-Json -Compress) }

$root = (Resolve-Path -LiteralPath $Producer).Path
$rootItem = Get-Item -LiteralPath $root -Force
Require ($rootItem.PSIsContainer) 'producer directory'
Require (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'real producer directory'
$manifestPath = Join-Path $root 'SOURCE_MANIFEST.json'
$manifestBytes = [IO.File]::ReadAllBytes($manifestPath)
$manifestHash = Get-LowerSha256 $manifestBytes
Require ($manifestHash -eq $ExpectedManifest) 'manifest external pin'
$manifest = [Text.Json.JsonSerializer]::Deserialize[Text.Json.JsonElement]($manifestBytes)
Require ($manifest.GetProperty('schema').GetString() -eq 'tactic-ramanujan384-scalable-source-manifest-v2') 'manifest schema'
Require ($manifest.GetProperty('source_root_sha256').GetString() -eq $ExpectedRoot) 'manifest root pin'

$names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$canonical = @()
foreach ($row in $manifest.GetProperty('members').EnumerateArray()) {
    $name = $row.GetProperty('name').GetString()
    Require ($names.Add($name)) "unique member $name"
    Require ([IO.Path]::GetFileName($name) -ceq $name) "flat member $name"
    $item = Get-Item -LiteralPath (Join-Path $root $name) -Force
    Require (-not $item.PSIsContainer) "file member $name"
    Require (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "non-link member $name"
    $bytes = [IO.File]::ReadAllBytes($item.FullName)
    $length = $row.GetProperty('bytes').GetInt64()
    $hash = $row.GetProperty('sha256').GetString()
    Require ($bytes.LongLength -eq $length) "length $name"
    Require ((Get-LowerSha256 $bytes) -eq $hash) "hash $name"
    $canonical += '{"bytes":' + $length + ',"name":' + (Json-String $name) + ',"sha256":' + (Json-String $hash) + '}'
}
Require ($canonical.Count -eq 14) 'fourteen producer members'
$actual = @((Get-ChildItem -LiteralPath $root -Force).Name | Sort-Object) -join "`n"
$wanted = @((@($names) + 'SOURCE_MANIFEST.json') | Sort-Object) -join "`n"
Require ($actual -ceq $wanted) 'exact flat producer closure'
$rootHash = Get-LowerSha256 ([Text.Encoding]::ASCII.GetBytes('[' + ($canonical -join ',') + ']'))
Require ($rootHash -eq $ExpectedRoot) 'independent canonical source root'

$verifierPath = (Resolve-Path -LiteralPath $ExternalVerifier).Path
$verifierBytes = [IO.File]::ReadAllBytes($verifierPath)
$verifierHash = Get-LowerSha256 $verifierBytes
Require ($verifierHash -eq $ExpectedVerifier) 'external verifier pin'
Require (-not $verifierPath.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) 'verifier outside package'

$core = [IO.File]::ReadAllText((Join-Path $root 'scalable_core.py'))
$adapter = [IO.File]::ReadAllText((Join-Path $root 'adapter.py'))
$runner = [IO.File]::ReadAllText((Join-Path $root 'run_source_free_cupy_target_fixture.py'))
$fixture = [IO.File]::ReadAllText((Join-Path $root 'source_free_fixture.py'))
$capability = [IO.File]::ReadAllText((Join-Path $root 'coarse_capability.py'))
$readme = [IO.File]::ReadAllText((Join-Path $root 'README.md'))
$verifier = [Text.Encoding]::UTF8.GetString($verifierBytes)

$checks = [ordered]@{
    splitmix_53bit_midpoint_box_muller = ($core.Contains('53-bit midpoint Box-Muller') -and $core.Contains('_splitmix64_numpy') -and $core.Contains('np.log(u1)') -and $core.Contains('np.cos(angle)') -and $core.Contains('np.sin(angle)'))
    canonical_f32_then_f64 = ($core.Contains('astype("<f4").astype("<f8")'))
    no_backend_rng_or_transcendentals = (-not $core.Contains('xp.random') -and -not $core.Contains('np.random'))
    cpu_cupy_byte_comparisons = ($runner.Contains('CPU/CuPy canonical Gaussian bytes') -and $runner.Contains('CPU/CuPy moment-matched Gaussian bytes'))
    one_batched_solve = (($core.Split('xp.linalg.solve(').Count - 1) -eq 1)
    all_candidate_einsum = $core.Contains('candidate_corrections = xp.einsum("bnk,brk->brn", atoms, dequantized)')
    no_item_sync = (-not $core.Contains('.item('))
    two_bulk_transfers_reported = $core.Contains('"bulk_device_to_host_transfers": 2 if hasattr(xp, "asnumpy") else 0')
    target_fixture_shape = ($fixture.Contains('intermediate: int = 128') -and $fixture.Contains('hidden: int = 2048'))
    all_controls_mandated = ($adapter.Contains('for seed in core.GAUSSIAN_SEEDS:') -and $adapter.Contains('_phase_control_host') -and $adapter.Contains('controls = _run_controls'))
    uint32_dimension_and_block_caps = ($core.Contains('positive uint32 intermediate') -and $core.Contains('positive uint32 hidden') -and $core.Contains('role block count exceeds inherited uint32 header'))
    uint64_length_and_rounding_caps = ($core.Contains('payload length exceeds inherited uint64 header') -and $core.Contains('page rounding overflow'))
    external_exact_closure = ($verifier.Contains('extra, missing, or nested package entry') -and $verifier.Contains('source root') -and -not $verifier.Contains('importlib'))
    coarse_source_receipt_and_output_hash_binding = ($capability.Contains('runtime coarse decoder source identity') -and $capability.Contains('independent coarse decoder PASS') -and $adapter.Contains('independently recorded literal coarse decode'))
    mechanism_only_claim_boundary = (($readme -match 'constructed periodic fixture validates mechanics only') -and ($readme -match 'neither a\s+Qwen result'))
}
foreach ($entry in $checks.GetEnumerator()) { Require ([bool]$entry.Value) "mechanism $($entry.Key)" }

$roleValues = [int64]128 * 2048
$weights = 3 * $roleValues
$coarseBytes = (307 * $weights) / 1024
$blocks = $roleValues / 4096
$fineBytes = 3 * $blocks * 48
$unpadded = 512 + $coarseBytes + $fineBytes
$physicalBytes = [int64]([math]::Ceiling($unpadded / 4096) * 4096)
$rate = 8.0 * $physicalBytes / $weights
Require ($coarseBytes -eq 235776 -and $fineBytes -eq 9216 -and $physicalBytes -eq 245760 -and $rate -eq 2.5) 'independent target-rate ledger'

$limits = [ordered]@{
    external_verify_to_runner_import_is_not_atomic = (-not $runner.Contains('source_root_sha256') -and $runner.Contains('spec.loader.exec_module(module)'))
    external_verifier_uses_path_reads_not_open_snapshot = (-not $verifier.Contains('os.open(') -and $verifier.Contains('path.read_bytes()'))
    live_decoder_instance_is_caller_supplied = ($capability.Contains('decoder: Any') -and $capability.Contains('type(decoder)'))
    live_decoder_instance_method_not_pinned = (-not $capability.Contains('decoder.__dict__') -and -not $capability.Contains('MethodType'))
    decoder_has_no_runtime_io_sandbox = (-not $capability.Contains('subprocess') -and -not $capability.Contains('wasm'))
    decoder_and_auditor_manifests_not_exact_closure_checked = (-not $capability.Contains('source_root_sha256') -and -not $capability.Contains('exact closure'))
    cupy_receipt_pending = ($readme -match 'CuPy target fixture pending')
}

$receipt = [ordered]@{
    schema = 'tactic-ramanujan384-v2-scalable-independent-static-review-receipt'
    producer_manifest_sha256 = $manifestHash
    producer_source_root_sha256 = $rootHash
    external_bootstrap_verifier_sha256 = $verifierHash
    producer_members = $canonical.Count
    exact_flat_closure_at_review_time = $true
    source_mechanism_checks = $checks
    independent_target_fixture_ledger = [ordered]@{ weights = $weights; coarse_bytes = $coarseBytes; fine_bytes = $fineBytes; physical_bytes = $physicalBytes; physical_rate_bpw = $rate }
    residual_limit_observations = $limits
    synthetic_10_bpw_is_mechanism_only = $true
    python_numpy_cupy_executed = $false
    qwen_payload_accessed = $false
    coarse_model_payload_accessed = $false
    network_accessed = $false
    disposition = $Disposition
}
$json = $receipt | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($Output, $json + "`n", [Text.UTF8Encoding]::new($false))
$json
