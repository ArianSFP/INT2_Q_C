[CmdletBinding()]
param(
    [string]$TargetRoot = (Join-Path $PSScriptRoot "..\lossy_tail_peeling_oracle_v8")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:CheckCount = 0

function Assert-Check {
    param([bool]$Condition, [string]$Message)
    $script:CheckCount++
    if (-not $Condition) { throw $Message }
}

function Get-Sha256Bytes {
    param([byte[]]$Bytes)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([Convert]::ToHexString($hasher.ComputeHash($Bytes))).ToLowerInvariant() }
    finally { $hasher.Dispose() }
}

function Get-FileRecord {
    param([string]$Path)
    Assert-Check (Test-Path -LiteralPath $Path -PathType Leaf) "missing file: $Path"
    $item = Get-Item -LiteralPath $Path -Force
    Assert-Check (-not (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) "reparse point rejected: $Path"
    $bytes = [IO.File]::ReadAllBytes($item.FullName)
    return [pscustomobject]@{ Bytes = $bytes; Length = [int64]$bytes.LongLength; Sha256 = Get-Sha256Bytes $bytes }
}

function Assert-NoDuplicateKeys {
    param([Text.Json.JsonElement]$Element, [string]$JsonPath)
    if ($Element.ValueKind -eq [Text.Json.JsonValueKind]::Object) {
        $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($property in $Element.EnumerateObject()) {
            Assert-Check ($seen.Add($property.Name)) "duplicate JSON key at $JsonPath.$($property.Name)"
            Assert-NoDuplicateKeys $property.Value "$JsonPath.$($property.Name)"
        }
    }
    elseif ($Element.ValueKind -eq [Text.Json.JsonValueKind]::Array) {
        $ordinal = 0
        foreach ($entry in $Element.EnumerateArray()) {
            Assert-NoDuplicateKeys $entry "$JsonPath[$ordinal]"
            $ordinal++
        }
    }
}

function Read-StrictJson {
    param([string]$Path)
    $record = Get-FileRecord $Path
    $encoding = [Text.UTF8Encoding]::new($false, $true)
    $text = $encoding.GetString($record.Bytes)
    $options = [Text.Json.JsonDocumentOptions]::new()
    $options.AllowTrailingCommas = $false
    $options.CommentHandling = [Text.Json.JsonCommentHandling]::Disallow
    $document = [Text.Json.JsonDocument]::Parse($text, $options)
    try { Assert-NoDuplicateKeys $document.RootElement '$' }
    finally { $document.Dispose() }
    $value = $text | ConvertFrom-Json -AsHashtable -Depth 100
    return [pscustomobject]@{ Text = $text; Value = $value; Record = $record }
}

function Convert-CanonicalObject {
    param([object]$Value, [string]$ExcludedRootField, [bool]$IsRoot = $false)
    if ($Value -is [Collections.IDictionary]) {
        $result = [ordered]@{}
        [string[]]$names = @($Value.Keys | ForEach-Object { [string]$_ })
        [Array]::Sort($names, [StringComparer]::Ordinal)
        foreach ($name in $names) {
            if ($IsRoot -and $name -ceq $ExcludedRootField) { continue }
            $result[$name] = Convert-CanonicalObject $Value[$name] $ExcludedRootField $false
        }
        return $result
    }
    if ($Value -is [Collections.IEnumerable] -and $Value -isnot [string]) {
        return @($Value | ForEach-Object { Convert-CanonicalObject $_ $ExcludedRootField $false })
    }
    return $Value
}

function Get-CanonicalSeal {
    param([Collections.IDictionary]$Value, [string]$Field)
    $canonical = Convert-CanonicalObject $Value $Field $true
    $text = ConvertTo-Json -InputObject $canonical -Compress -Depth 100
    return Get-Sha256Bytes ([Text.UTF8Encoding]::new($false, $true).GetBytes($text))
}

function Assert-CanonicalSeal {
    param([pscustomobject]$Parsed, [string]$Field, [string]$Label)
    $claimed = $Parsed.Value[$Field]
    Assert-Check ($claimed -is [string] -and $claimed -match '^[0-9a-f]{64}$') "$Label seal grammar"
    Assert-Check ((Get-CanonicalSeal $Parsed.Value $Field) -ceq $claimed) "$Label canonical seal mismatch"
}

function Assert-ExactSet {
    param([object[]]$Actual, [string[]]$Expected, [string]$Label)
    $set = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($entry in $Actual) { Assert-Check ($entry -is [string] -and $set.Add([string]$entry)) "$Label duplicate/non-string" }
    $expectedSet = [Collections.Generic.HashSet[string]]::new($Expected, [StringComparer]::Ordinal)
    Assert-Check ($set.SetEquals($expectedSet) -and $set.Count -eq $Expected.Count) "$Label mismatch"
}

function Get-TextFile {
    param([string]$Path)
    $record = Get-FileRecord $Path
    return ([Text.UTF8Encoding]::new($false, $true)).GetString($record.Bytes)
}

function Assert-Contains {
    param([string]$Text, [string]$Needle, [string]$Label)
    Assert-Check ($Text.Contains($Needle, [StringComparison]::Ordinal)) "$Label missing: $Needle"
}

function Get-DecisionModel {
    param([double]$Optimistic, [double]$FiniteAbsolute, [double]$FiniteCalibrated)
    foreach ($value in @($Optimistic, $FiniteAbsolute, $FiniteCalibrated)) {
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { throw "non-finite decision input" }
    }
    $target = -0.5 * [Math]::Log(0.8, 2.0)
    $kill = $target - 0.02
    $guard = 0.0001
    $distances = @(
        [Math]::Abs($Optimistic - $kill), [Math]::Abs($Optimistic - $target),
        [Math]::Abs($FiniteAbsolute - $kill), [Math]::Abs($FiniteAbsolute - $target),
        [Math]::Abs($FiniteCalibrated - $kill), [Math]::Abs($FiniteCalibrated - $target)
    )
    if (@($distances | Where-Object { $_ -le $guard }).Count -gt 0) { return "HOLD_NUMERIC_BOUNDARY" }
    if ($FiniteAbsolute -ge $target -and $FiniteCalibrated -ge $target) { return "FINITE_CODEC_WARRANTED" }
    if ($Optimistic -lt $kill) { return "EARLY_KILL_FAR_SHORT" }
    if ($Optimistic -lt $target) { return "HOLD_OPTIMISTIC_NEAR_BOUNDARY" }
    return "OPTIMISTIC_SURVIVOR"
}

$auditRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$targetRoot = [IO.Path]::GetFullPath($TargetRoot)
$manifest = Read-StrictJson (Join-Path $auditRoot "audit_manifest.json")
$receipt = Read-StrictJson (Join-Path $auditRoot "audit_receipt.json")
$replay = Read-StrictJson (Join-Path $auditRoot "replay_receipt.json")
Assert-CanonicalSeal $manifest "audit_manifest_sha256" "audit manifest"
Assert-CanonicalSeal $receipt "audit_receipt_sha256" "audit receipt"
Assert-CanonicalSeal $replay "replay_receipt_sha256" "replay receipt"
Assert-Check ($manifest.Value["schema"] -ceq "lossy-tail-v8-independent-source-audit-manifest-v1") "audit manifest schema"
Assert-Check ($manifest.Value["status"] -ceq "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET") "audit manifest status"
Assert-Check ($receipt.Value["schema"] -ceq "lossy-tail-v8-independent-source-audit-receipt-v1") "audit receipt schema"
Assert-Check ($receipt.Value["status"] -ceq "PASS_V8_INDEPENDENT_SOURCE_AUDIT") "audit receipt status"
Assert-Check ($replay.Value["status"] -ceq "PASS_DOCUMENTED_SOURCE_ONLY_REPLAY") "replay status"

$auditArtifacts = @($manifest.Value["audit_artifacts"])
$expectedAuditArtifacts = @("README.md", "audit_receipt.json", "replay_receipt.json", "verify_audit.ps1")
Assert-ExactSet @($auditArtifacts | ForEach-Object { $_["path"] }) $expectedAuditArtifacts "audit artifacts"
foreach ($row in $auditArtifacts) {
    Assert-ExactSet @($row.Keys) @("path", "bytes", "sha256") "audit artifact row keys"
    $record = Get-FileRecord (Join-Path $auditRoot $row["path"])
    Assert-Check ($record.Length -eq [int64]$row["bytes"]) "audit artifact bytes: $($row['path'])"
    Assert-Check ($record.Sha256 -ceq [string]$row["sha256"]) "audit artifact hash: $($row['path'])"
}

$packageRows = @($replay.Value["tested_package_members"])
$expectedPackage = @(
    "ARTIFACT_HASHES.json", "audit_lock_entrypoint.py", "authorization_contract.json",
    "CPU_TEST_RECEIPT.json", "launch_manifest.json", "lossy_tail_core.py",
    "lossy_tail_oracle.py", "preflight_launch.py", "protocol_lock.json", "README.md",
    "repair_lock.json", "runtime_calibrate.py", "runtime_contract.json",
    "source_bindings.json", "test_lossy_tail_core.py", "test_release_security.py",
    "verify_package.py"
)
Assert-Check ($packageRows.Count -eq 17) "package row count"
Assert-ExactSet @($packageRows | ForEach-Object { $_["path"] }) $expectedPackage "package rows"
$observedTarget = @(Get-ChildItem -LiteralPath $targetRoot -Force)
Assert-Check ($observedTarget.Count -eq 17) "target package closure count"
Assert-Check (@($observedTarget | Where-Object { $_.PSIsContainer -or ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) }).Count -eq 0) "target package contains non-regular entry"
Assert-ExactSet @($observedTarget | ForEach-Object { $_.Name }) $expectedPackage "target package closure"
foreach ($row in $packageRows) {
    Assert-ExactSet @($row.Keys) @("path", "bytes", "sha256") "package row keys"
    $record = Get-FileRecord (Join-Path $targetRoot $row["path"])
    Assert-Check ($record.Length -eq [int64]$row["bytes"]) "target bytes: $($row['path'])"
    Assert-Check ($record.Sha256 -ceq [string]$row["sha256"]) "target SHA-256: $($row['path'])"
}

$stageRows = @($manifest.Value["target_stage_members"])
$expectedStage = @(
    "authorization_contract.json", "audit_lock_entrypoint.py", "launch_manifest.json",
    "lossy_tail_core.py", "lossy_tail_oracle.py", "preflight_launch.py",
    "protocol_lock.json", "repair_lock.json", "runtime_calibrate.py",
    "runtime_contract.json", "source_bindings.json"
)
Assert-Check ($stageRows.Count -eq 11) "stage row count"
Assert-ExactSet @($stageRows | ForEach-Object { $_["path"] }) $expectedStage "stage rows"
foreach ($row in $stageRows) {
    $record = Get-FileRecord (Join-Path $targetRoot $row["path"])
    Assert-Check ($record.Length -eq [int64]$row["bytes"]) "stage bytes: $($row['path'])"
    Assert-Check ($record.Sha256 -ceq [string]$row["sha256"]) "stage SHA-256: $($row['path'])"
}

$launch = Read-StrictJson (Join-Path $targetRoot "launch_manifest.json")
Assert-Check ($launch.Record.Sha256 -ceq "6c5f5cd05973dbc0bf16cd9ea39951e690b15e15e13e969d2a33823117c2aa94") "launch SHA-256"
Assert-Check ($launch.Value["schema"] -ceq "lossy-tail-v8-launch-manifest-v1") "launch schema"
Assert-Check ($launch.Value["status"] -ceq "FROZEN_V8_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION") "launch status"
Assert-ExactSet @($launch.Value["allowed_members"]) $expectedStage "launch allowed members"
$launchRows = @($launch.Value["members"])
Assert-Check ($launchRows.Count -eq 10) "launch member row count"
Assert-ExactSet @($launchRows | ForEach-Object { $_["path"] }) @($expectedStage | Where-Object { $_ -cne "launch_manifest.json" }) "launch member rows"
foreach ($row in $launchRows) {
    $record = Get-FileRecord (Join-Path $targetRoot $row["path"])
    Assert-Check ($record.Length -eq [int64]$row["bytes"] -and $record.Sha256 -ceq [string]$row["sha256"]) "launch identity: $($row['path'])"
}

$artifact = Read-StrictJson (Join-Path $targetRoot "ARTIFACT_HASHES.json")
Assert-CanonicalSeal $artifact "artifact_manifest_sha256" "producer artifact manifest"
Assert-Check ($artifact.Record.Sha256 -ceq "c71ff25df401188b0557e859e5b105ec274f25e6411ce7c24177fd348734c6b2") "artifact file SHA-256"
Assert-Check (@($artifact.Value["members"]).Count -eq 16) "artifact member row count"
$repair = Read-StrictJson (Join-Path $targetRoot "repair_lock.json")
Assert-CanonicalSeal $repair "repair_lock_sha256" "repair lock"
Assert-Check ($repair.Value["repair_lock_sha256"] -ceq "270746d99395b9c713500dad8b9c41d8c93aa58ee0d0e04adb815da44870fd32") "repair internal SHA-256"

$protocol = Read-StrictJson (Join-Path $targetRoot "protocol_lock.json")
$runtimeContract = Read-StrictJson (Join-Path $targetRoot "runtime_contract.json")
$authorizationContract = Read-StrictJson (Join-Path $targetRoot "authorization_contract.json")
$core = Get-TextFile (Join-Path $targetRoot "lossy_tail_core.py")
$calibrator = Get-TextFile (Join-Path $targetRoot "runtime_calibrate.py")
Assert-Check ($protocol.Value["status"] -ceq "FROZEN_V8_BEFORE_ANY_RUNTIME_CALIBRATION_PAYLOAD_OR_GPU_EXECUTION") "protocol status"
Assert-Check ($runtimeContract.Value["status"] -ceq "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION") "runtime contract status"
Assert-Check ($authorizationContract.Value["status"] -ceq "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS") "authorization template status"
Assert-Contains $calibrator 'flags = ["--manifest", "--manifest-sha256", "--output"]' "calibrator grammar"
Assert-Contains $calibrator 'probe = oracle_module.runtime_probe(cp)' "calibrator runtime-probe call"
Assert-Check (-not $calibrator.Contains('build_panel(', [StringComparison]::Ordinal)) "calibrator calls Qwen panel builder"
Assert-Check (-not $calibrator.Contains('load_qwen_pair(', [StringComparison]::Ordinal)) "calibrator calls payload loader"
Assert-Contains $calibrator '"status": "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT"' "untrusted runtime receipt status"
Assert-Contains $calibrator '"model_or_qwen_paths_supplied": 0' "source-free supplied-path ledger"
Assert-Contains $calibrator '"payload_files_opened": 0' "source-free payload ledger"

$matrixN = [int64]768 * 2048
$panelN = [int64]12 * $matrixN
Assert-Check ($matrixN -eq 1572864 -and $panelN -eq 18874368) "panel dimensions"
$targetS = -0.5 * [Math]::Log(0.8, 2.0)
$killS = $targetS - 0.02
Assert-Check ([Math]::Abs($targetS - 0.16096404744368115) -lt 1e-16) "target recomputation"
Assert-Check ([Math]::Abs([Math]::Pow(2.0, -2.0 * $targetS) - 0.8) -lt 2e-16) "target roundtrip"
Assert-Check ([Math]::Abs($killS - [double]$protocol.Value["target"]["optimistic_kill_threshold_s_bpw"]) -lt 1e-15) "kill threshold recomputation"
Assert-Check ([Math]::Abs($killS - [double]$protocol.Value["target"]["optimistic_kill_threshold_s_bpw"]) -lt 0.0001) "kill threshold inside numeric guard"

$expectedRates = @(
    [pscustomobject]@{ Requested = 2.15; Bytes = 5072486; Bits = 40579888; Actual = 2.1499998304578991 },
    [pscustomobject]@{ Requested = 2.30; Bytes = 5426380; Bits = 43411040; Actual = 2.2999996609157987 },
    [pscustomobject]@{ Requested = 2.50; Bytes = 5898240; Bits = 47185920; Actual = 2.5 }
)
foreach ($rate in $expectedRates) {
    $bytes = [int64][Math]::Floor($rate.Requested * $panelN / 8.0)
    $bits = 8 * $bytes
    $actual = [double]$bits / $panelN
    Assert-Check ($bytes -eq $rate.Bytes -and $bits -eq $rate.Bits) "physical capacity recomputation: $($rate.Requested)"
    Assert-Check ([Math]::Abs($actual - $rate.Actual) -lt 1e-15) "physical rate recomputation: $($rate.Requested)"
    $commonBits = 4096 * 8 + 144 * 8
    $sideByExpert = @(640, 712, 784, 856, 928, 1000)
    $payload = $bits - $commonBits - ($sideByExpert | Measure-Object -Sum).Sum - 7 * 6
    Assert-Check ($payload -gt 0) "positive independent payload fixture"
    $base = [Math]::Floor($payload / 6)
    $payloadByExpert = @($base, $base, $base, $base, $base, ($payload - 5 * $base))
    $frameBits = for ($i = 0; $i -lt 6; $i++) { $u = [int64]$sideByExpert[$i] + [int64]$payloadByExpert[$i]; $u + ((-$u) % 8 + 8) % 8 }
    $closure = [int64]$commonBits + [int64](($frameBits | Measure-Object -Sum).Sum)
    $trailer = $bits - $closure
    Assert-Check ($trailer -ge 0 -and $trailer -le 42 -and $closure + $trailer -eq $bits) "rate ledger closure: $($rate.Requested)"
}
Assert-Contains $core 'mse = total_sse / source_energy' "relative-MSE formula"
Assert-Contains $core 'actual_rate = physical_bits / PANEL_N' "physical-rate formula"
Assert-Contains $core 'f_value = mse * 2.0 ** (2.0 * actual_rate)' "F formula"
Assert-Contains $core 'maximum_logical < 2.0 and maximum_page < 2.0' "strict read maxima"
Assert-Contains $core 'expert["cold_page_amplification"], "expert page") < 2.0' "strict expert page gate"
Assert-Check (1.999999 -lt 2.0 -and -not (2.0 -lt 2.0)) "strict read boundary model"

$meanTolerance = 64 * [Math]::Pow(2.0, -23) * [Math]::Max([Math]::Sqrt(0.0004), [Math]::Max([Math]::Abs(0.01), [Math]::Pow(2.0, -126)))
$varianceTolerance = 256 * [Math]::Pow(2.0, -23) * [Math]::Max(0.0004, [Math]::Max(0.0001, [Math]::Pow(2.0, -252)))
Assert-Check ([Math]::Abs($meanTolerance - 1.52587890625e-7) -lt 1e-22) "mean tolerance fixture"
Assert-Check ([Math]::Abs($varianceTolerance - 1.220703125e-8) -lt 1e-22) "variance tolerance fixture"
Assert-Check (0.99 * $meanTolerance -le $meanTolerance -and 1.01 * $meanTolerance -gt $meanTolerance) "moment tolerance boundary fixture"
Assert-Contains $core 'MEAN_TOLERANCE_ULPS * FLOAT32_EPSILON * mean_scale' "mean tolerance source formula"
Assert-Contains $core 'VARIANCE_TOLERANCE_ULPS * FLOAT32_EPSILON * variance_scale' "variance tolerance source formula"
Assert-Contains $core 'require_finite_tree(row, f"decision row[{ordinal}]")' "recursive finite decision gate"
Assert-Contains $core 'finite_joint > optimistic_m + DECISION_CONSISTENCY_EPSILON_S' "finite envelope consistency"
Assert-Contains $core 'if ".both_axis_" in component.name:' "live support-XKLT angle charge"
Assert-Check ((Get-DecisionModel 0.14 0.10 0.10) -ceq "EARLY_KILL_FAR_SHORT") "early-kill decision model"
Assert-Check ((Get-DecisionModel $killS 0.10 0.10) -ceq "HOLD_NUMERIC_BOUNDARY") "numeric-boundary decision model"
Assert-Check ((Get-DecisionModel 0.15 0.13 0.13) -ceq "HOLD_OPTIMISTIC_NEAR_BOUNDARY") "near-boundary decision model"
Assert-Check ((Get-DecisionModel 0.18 0.17 0.17) -ceq "FINITE_CODEC_WARRANTED") "finite promotion decision model"
Assert-Check ((Get-DecisionModel 0.18 0.13 0.13) -ceq "OPTIMISTIC_SURVIVOR") "optimistic survivor decision model"
$nonfiniteRejected = $false
try { Get-DecisionModel ([double]::NaN) 0.2 0.2 | Out-Null } catch { $nonfiniteRejected = $true }
Assert-Check $nonfiniteRejected "non-finite decision model rejection"

$testTotals = $replay.Value["test_totals"]
Assert-Check ([int]$testTotals["unique_test_cases"] -eq 33 -and [int]$testTotals["unique_test_cases_passed"] -eq 33) "unique test totals"
Assert-Check ([int]$testTotals["total_test_case_executions"] -eq 36 -and [int]$testTotals["total_passed_executions"] -eq 36) "execution totals"
Assert-Check ([int]$testTotals["failures"] -eq 0 -and [int]$testTotals["errors"] -eq 0 -and [int]$testTotals["skipped"] -eq 0) "clean test totals"
Assert-Check ($receipt.Value["verdict"]["source_free_runtime_calibration_warranted"] -eq $true) "calibration verdict"
Assert-Check ($receipt.Value["verdict"]["payload_access_authorized"] -eq $false -and $receipt.Value["verdict"]["production_authorized"] -eq $false) "authority boundary"
foreach ($key in @(
    "model_or_qwen_paths_traversed", "model_payload_files_opened", "validation_data_files_opened",
    "production_result_files_opened", "torch_imports", "cupy_imports", "cuda_initializations",
    "gpu_jobs", "runtime_calibrations", "production_runs", "production_authorizations_created",
    "external_data_network_calls"
)) {
    Assert-Check ([int64]$receipt.Value["access_ledger"][$key] -eq 0) "nonzero access ledger: $key"
}
Assert-Check ([int]$receipt.Value["check_summary"]["blocker_count"] -eq 0) "unexpected blocker"
Assert-Check ($receipt.Value["authorization"] -ceq "PASS WARRANTS ONLY A SEPARATELY APPROVED SOURCE-FREE RUNTIME CALIBRATION; NO MODEL/PAYLOAD/GPU/PRODUCTION AUTHORITY IS CREATED BY THIS AUDIT") "authorization boundary text"

$expectedCount = [int]$receipt.Value["check_summary"]["independent_check_count"]
Assert-Check ($expectedCount -eq ($script:CheckCount + 1)) "independent check-count binding"
$result = [ordered]@{
    verifier = "lossy-tail-v8-independent-source-audit-verifier-v1"
    status = "PASS_V8_INDEPENDENT_SOURCE_AUDIT"
    audited_launch_manifest_sha256 = $launch.Record.Sha256
    audit_manifest_file_sha256 = $manifest.Record.Sha256
    audit_manifest_internal_sha256 = $manifest.Value["audit_manifest_sha256"]
    audit_receipt_file_sha256 = $receipt.Record.Sha256
    audit_receipt_internal_sha256 = $receipt.Value["audit_receipt_sha256"]
    unique_tests_passed = 33
    test_case_executions_passed = 36
    independent_checks = $script:CheckCount
    blockers = 0
    model_payload_files_opened = 0
    cupy_imports = 0
    cuda_initializations = 0
    gpu_jobs = 0
    payload_access_authorized = $false
    source_free_runtime_calibration_warranted = $true
}
Write-Output ($result | ConvertTo-Json -Compress)
