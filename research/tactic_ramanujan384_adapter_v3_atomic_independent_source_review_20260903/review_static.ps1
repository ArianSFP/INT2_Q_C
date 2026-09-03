param(
    [string]$Producer = (Join-Path $PSScriptRoot '..\tactic_ramanujan384_adapter_v3_atomic'),
    [string]$Bootstrap = (Join-Path $PSScriptRoot '..\tactic_ramanujan384_adapter_v3_atomic_bootstrap.py')
)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1'
$ExpectedRoot = '5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674'
$ExpectedBootstrap = 'f7e8cd469b0ff9dd9ef09b400c63ec9f91e067f849d6b009588ea94ad6494375'
$ExpectedV2Manifest = '1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209'
$ExpectedV2Root = 'bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495'
$ExpectedReviewManifest = '4ed8c0fe24db072e22aef84791a01ccf637cb337376a389d47119248fd257281'
$ExpectedReviewRoot = '16ea8dfde5cf7a48552dc7b5a74b209488934b8764e890bf51bb5cd02985cd39'

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "atomic-v3 static review failed: $Message" }
}
function Hash-Bytes([byte[]]$Bytes) {
    [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}
function Check-Closure([string]$Directory, [string]$ManifestName,
                       [string]$ManifestHash, [string]$RootHash) {
    $Root = (Resolve-Path -LiteralPath $Directory).Path
    $ManifestPath = Join-Path $Root $ManifestName
    $ManifestBytes = [IO.File]::ReadAllBytes($ManifestPath)
    Require ((Hash-Bytes $ManifestBytes) -eq $ManifestHash) "$Directory manifest"
    $Manifest = [Text.Encoding]::UTF8.GetString($ManifestBytes) | ConvertFrom-Json
    $Rows = @()
    foreach ($Row in $Manifest.members) {
        $Bytes = [IO.File]::ReadAllBytes((Join-Path $Root $Row.name))
        $Digest = Hash-Bytes $Bytes
        Require ($Bytes.Length -eq $Row.bytes -and $Digest -eq $Row.sha256) "$Directory member $($Row.name)"
        $Rows += [ordered]@{bytes=[int64]$Bytes.Length;name=[string]$Row.name;sha256=$Digest}
    }
    $Canonical = ConvertTo-Json -InputObject $Rows -Compress -Depth 5
    $Computed = Hash-Bytes ([Text.Encoding]::ASCII.GetBytes($Canonical))
    Require ($Computed -eq $RootHash -and $Manifest.source_root_sha256 -eq $RootHash) "$Directory root"
    $Entries = @(Get-ChildItem -Force -LiteralPath $Root)
    Require ($Entries.Count -eq $Rows.Count + 1 -and @($Entries | Where-Object {$_.PSIsContainer}).Count -eq 0) "$Directory exact flat closure"
    return [pscustomobject]@{Root=$Root;Manifest=$Manifest;Members=$Rows.Count}
}

$V3 = Check-Closure $Producer 'SOURCE_MANIFEST.json' $ExpectedManifest $ExpectedRoot
$Research = Split-Path -Parent $V3.Root
$V2 = Check-Closure (Join-Path $Research 'tactic_ramanujan384_adapter_v2_scalable') 'SOURCE_MANIFEST.json' $ExpectedV2Manifest $ExpectedV2Root
$V2Review = Check-Closure (Join-Path $Research 'tactic_ramanujan384_adapter_v2_scalable_independent_source_review_20260903') 'source_manifest.json' $ExpectedReviewManifest $ExpectedReviewRoot
$BootstrapBytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Bootstrap))
Require ((Hash-Bytes $BootstrapBytes) -eq $ExpectedBootstrap) 'external bootstrap pin'

$BootstrapSource = [Text.Encoding]::UTF8.GetString($BootstrapBytes)
$Runner = Get-Content -Raw -LiteralPath (Join-Path $V3.Root 'snapshot_runner.py')
$Adapter = Get-Content -Raw -LiteralPath (Join-Path $V3.Root 'adapter_atomic.py')
$Worker = Get-Content -Raw -LiteralPath (Join-Path $V3.Root 'coarse_byte_worker.py')
$Core = Get-Content -Raw -LiteralPath (Join-Path $V2.Root 'scalable_core.py')

Require ($BootstrapSource.Contains('os.open(') -and $BootstrapSource.Contains('os.fstat(descriptor)') -and $BootstrapSource.Contains('before_entries == expected_entries == after_entries')) 'descriptor reads and double closure'
Require ($BootstrapSource.Contains('write_snapshot(snapshot_root, combined)') -and $BootstrapSource.Contains('immutable = verify_snapshot(snapshot_root, combined)') -and $BootstrapSource.Contains('return MappingProxyType(output)')) 'verified mapping-proxy snapshot'
Require ($BootstrapSource.IndexOf('immutable = verify_snapshot') -lt
         $BootstrapSource.IndexOf('return execute_runner')) 'snapshot precedes runner'
Require ($Runner.Contains('compile(snapshot[key]') -and
         $Runner.Contains('isinstance(snapshot_bytes, MappingProxyType)') -and
         -not $Runner.Contains('spec_from_file_location') -and
         -not $Runner.Contains('read_bytes')) 'runner snapshot-only project loader'

Require (-not $Adapter.Contains('decoder: Any') -and
         $Adapter.Contains('no live decoder capability') -and
         $Adapter.Contains('mutable_decoder_object_used') -and
         $Adapter.Contains('coarse_worker_api.authenticate_and_decode')) 'no live decoder object'
Require ($Worker.Contains('program["imports"] == []') -and
         $Worker.Contains('program["opcode"] == "ZERO_F32_LE"') -and
         $Worker.Contains('worker byte buffers') -and
         $Worker.Contains('program exact schema')) 'zero-import pathless program'
Require ($Worker.Contains('capability external closure pins') -and
         $Worker.Contains('independent audit receipt external pin') -and
         $Worker.Contains('output_f32_sha256_by_role') -and
         $Worker.Contains('exact closure entries')) 'worker/auditor closure and receipt pins'

Require ($Core.Contains('MAX_RANK = 14') -and
         $Core.Contains('candidate_ranks_batched') -and
         $Core.Contains('per_candidate_host_scalar_syncs') -and
         $Adapter.Contains('for seed in core.GAUSSIAN_SEEDS') -and
         $Adapter.Contains('controls = _run_controls') -and
         $Adapter.Contains('if not 2.15 <= physical_rate <= 2.5') -and
         $Adapter.Contains('core.MIN_CONTROL_EXCESS_BPW')) 'v2 batched control/rate semantics'
$ShapeValues = 3 * 128 * 2048
$PhysicalBytes = 245760
Require ((8.0 * $PhysicalBytes / $ShapeValues) -eq 2.5) 'independent target ledger'

$OwnRead = $BootstrapSource.IndexOf('own_payload = safe_read(Path(__file__)')
$FirstImport = $BootstrapSource.IndexOf('import argparse')
Require ($FirstImport -ge 0 -and $FirstImport -lt $OwnRead) 'bootstrap code/imports precede self-hash'
Require (-not $BootstrapSource.Contains('sys.flags.isolated') -and
         -not $BootstrapSource.Contains('PYTHONPATH')) 'bootstrap isolation not enforced'
Require ($Worker.Contains('all(isinstance(value, str) and len(value) == 64 for value in pins)') -and
         -not $Worker.Contains('len(set(pins))')) 'pin uniqueness not enforced'
Require (-not $Worker.Contains('worker_source_directory != auditor_source_directory') -and -not $Worker.Contains('expected_worker_source_root_sha256 != expected_auditor_source_root_sha256')) 'worker/auditor non-alias not enforced'
Require ($Worker.Contains('program["opcode"] == "ZERO_F32_LE"') -and
         -not $Worker.Contains('elif program["opcode"]')) 'fixture-only zero opcode'

[ordered]@{
    schema='tactic-ramanujan384-v3-atomic-independent-static-powershell-receipt'
    producer_manifest_sha256=$ExpectedManifest
    producer_source_root_sha256=$ExpectedRoot
    external_bootstrap_sha256=$ExpectedBootstrap
    producer_members=$V3.Members
    exact_atomic_snapshot_mechanism=$true
    immutable_mapping_proxy_compilation=$true
    live_decoder_object_absent=$true
    zero_import_no_path_byte_program=$true
    v2_batched_controls_and_rate_semantics_preserved=$true
    bootstrap_preexecution_authentication_source_enforced=$false
    worker_auditor_nonalias_source_enforced=$false
    real_coarse_decoder_present=$false
    python_or_cupy_executed=$false
    payloads_opened=0
    status='PASS_STATIC_REPAIRS__CONDITIONAL_BOOTSTRAP_AND_INDEPENDENT_WORKER_TRUST__RUNTIME_PAYLOAD_HELD'
} | ConvertTo-Json -Depth 5
