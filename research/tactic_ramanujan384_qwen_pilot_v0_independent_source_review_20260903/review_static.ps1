param([Parameter(Mandatory = $true)][string]$Producer)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '340ef7f532ab02e03bf04257f3ff07dbc4736bd9e5e96203169603df918e3a8a'
$ExpectedRoot = '611bf1b9c822cb90f32a2956e52d8332ef75374186e4acedc958ec3a6c5468ec'
$ExpectedMembers = @(
    'README.md', 'SOURCE_ONLY_TEST_RESULT.json', 'aperture.py', 'capability.py',
    'design_lock.json', 'pilot_runner.py', 'test_source_only.py', 'verify_source.py'
)

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Hash-Lower([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}
function Require-Contains([string]$Text, [string]$Needle, [string]$Label) {
    Require ($Text.Contains($Needle)) $Label
}

$Root = (Resolve-Path -LiteralPath $Producer).Path
$ManifestPath = Join-Path $Root 'SOURCE_MANIFEST.json'
Require ((Hash-Lower $ManifestPath) -eq $ExpectedManifest) 'producer manifest hash'
$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
Require ($Manifest.schema -eq 'tactic-ramanujan384-qwen-pilot-v0-source-manifest') 'schema'
Require ($Manifest.status -eq 'FROZEN_SOURCE_ONLY__COMPILE_TIME_CAPABILITY_PIN_NONE__NO_PAYLOAD_OPENED') 'status'
Require ($Manifest.source_root_sha256 -eq $ExpectedRoot) 'root pin'
Require ($Manifest.members.Count -eq 8) 'member count'

$Observed = [System.Collections.Generic.List[string]]::new()
$Names = [System.Collections.Generic.List[string]]::new()
foreach ($Row in $Manifest.members) {
    $Name = [string]$Row.name
    Require ($Name -eq [IO.Path]::GetFileName($Name)) 'flat member name'
    Require (-not $Names.Contains($Name)) 'unique member name'
    $Path = Join-Path $Root $Name
    $Item = Get-Item -LiteralPath $Path -Force
    Require (-not $Item.PSIsContainer -and -not $Item.LinkType) 'regular non-link member'
    Require ($Item.Length -eq [Int64]$Row.bytes) "member size $Name"
    Require ((Hash-Lower $Path) -eq [string]$Row.sha256) "member hash $Name"
    $Bytes = [Convert]::ToString([Int64]$Row.bytes, [Globalization.CultureInfo]::InvariantCulture)
    $Observed.Add('{"bytes":' + $Bytes + ',"name":"' + $Name + '","sha256":"' + [string]$Row.sha256 + '"}')
    $Names.Add($Name)
}
$Sorted = @($Names | Sort-Object -CaseSensitive)
Require ((Compare-Object @($Names) $Sorted).Count -eq 0) 'canonical member order'
$Entries = @(Get-ChildItem -LiteralPath $Root -Force | ForEach-Object Name | Sort-Object -CaseSensitive)
$ExpectedEntries = @($ExpectedMembers + 'SOURCE_MANIFEST.json' | Sort-Object -CaseSensitive)
Require ((Compare-Object $Entries $ExpectedEntries).Count -eq 0) 'exact closure'
$CanonicalRows = '[' + ($Observed -join ',') + ']'
$Hasher = [Security.Cryptography.SHA256]::Create()
$RootHash = -join ($Hasher.ComputeHash([Text.Encoding]::ASCII.GetBytes($CanonicalRows)) |
    ForEach-Object { $_.ToString('x2') })
Require ($RootHash -eq $ExpectedRoot) 'canonical source root'

$Capability = Get-Content -Raw -LiteralPath (Join-Path $Root 'capability.py')
$Aperture = Get-Content -Raw -LiteralPath (Join-Path $Root 'aperture.py')
$Runner = Get-Content -Raw -LiteralPath (Join-Path $Root 'pilot_runner.py')
$Readme = Get-Content -Raw -LiteralPath (Join-Path $Root 'README.md')
$TestReceipt = Get-Content -Raw -LiteralPath (Join-Path $Root 'SOURCE_ONLY_TEST_RESULT.json') | ConvertFrom-Json

Require-Contains $Capability 'TRUSTED_CAPABILITY_SHA256: str | None = None' 'compile-time hold'
Require-Contains $Capability 'HOLD: compile-time external capability SHA-256 is None' 'hold before path'
Require-Contains $Capability 'manifest_name": "source_manifest.json"' 'legacy lowercase manifest'
Require-Contains $Capability 'root_row_order": "name_bytes_sha256"' 'legacy row algorithm'
Require-Contains $Capability 'source_snapshot_root_sha256' 'domain-root field'
Require-Contains $Capability '5441435449432d41435455414c2d434f415253452d4e31382d56362d524553554c' 'domain bytes'
Require-Contains $Capability 'runtime receipt is a pinned package member' 'receipt membership'
Require-Contains $Capability 'audited_manifest_sha256' 'audit manifest binding'
Require-Contains $Capability 'audited_source_root_sha256' 'audit root binding'
Require-Contains $Capability 'runtime audit package identity alias' 'audit anti-alias'

Require-Contains $Aperture 'tuple(range(MAX_RANK + 1))' 'ranks 0..14'
Require-Contains $Aperture 'core.encode_packet' 'literal encode'
Require-Contains $Aperture 'core.decode_packet' 'literal decode'
Require-Contains $Aperture 'candidate_sse_device = xp.where' 'invalid-state exclusion'
Require-Contains $Aperture 'batched_solve_calls' 'batched solve'
Require-Contains $Aperture 'min(owner_lcb.values()) >= REQUIRED_CAPTURE' 'owner LCB gate'
Require-Contains $Aperture 'conservative_d <= TARGET_D' 'D LCB gate'

$Authorize = $Runner.IndexOf('capability.authorize_production')
$Cupy = $Runner.IndexOf('importlib.import_module("cupy")')
$EarlyKill = $Runner.IndexOf('HARD_KILL_SOURCE_FIRST_APERTURE')
$Full = $Runner.IndexOf('core.encode_role_batched', $EarlyKill)
$Controls = $Runner.IndexOf('controls = run_controls')
$FullKill = $Runner.IndexOf('HARD_KILL_FULL_EXPERT_D_GT_0_025')
Require ($Authorize -ge 0 -and $Cupy -gt $Authorize) 'authority precedes CuPy'
Require ($EarlyKill -ge 0 -and $Full -ge 0 -and $EarlyKill -lt $Full) 'early kill precedes full search'
Require ($FullKill -ge 0 -and $Controls -gt $FullKill) 'D kill precedes controls'
Require-Contains $Runner 'decoded["coarse"] == authenticated["coarse_bytes"]' 'decoded coarse equality'
Require-Contains $Runner '8 * len(composite) * capability.EXPECTED_RATE_DENOMINATOR' 'rate equation'
Require-Contains $Runner 'independent_decoded_score' 'FP64 source rescore'
Require-Contains $Runner 'one_pass_page_trace' 'host page trace'
Require-Contains $Runner '"projected_transfer_used": False' 'projection rejection'
Require-Contains $Readme 'RUN_PINNED_TACTIC_RAMANUJAN384_QWEN_PILOT_V0' 'RunPod command'
Require ($TestReceipt.final_frozen_source_tests_executed -eq $false) 'honest test hold'
Require ($TestReceipt.pre_hardening_execution.tests_run -eq 17) 'historical tests'
Require ($TestReceipt.production_authorized -eq $false) 'production hold'

[ordered]@{
    schema = 'tactic-ramanujan384-qwen-pilot-v0-independent-static-replay-v1'
    status = 'PASS_STATIC_SOURCE_ARCHITECTURE_RUNTIME_HELD'
    producer_manifest_sha256 = $ExpectedManifest
    producer_source_root_sha256 = $ExpectedRoot
    members = 8
    final_frozen_python_tests_executed = $false
    payload_accessed = $false
    disposition = 'PASS_STATIC_FAIL_CLOSED_QWEN_PILOT_ARCHITECTURE__HOLD_FINAL_SOURCE_PYTHON_CUPY_CAPABILITY_PAYLOAD_RD_AND_HBM'
} | ConvertTo-Json -Depth 5
