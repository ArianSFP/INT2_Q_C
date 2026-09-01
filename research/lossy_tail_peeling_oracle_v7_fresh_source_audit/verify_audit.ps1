[CmdletBinding()]
param(
    [string]$TargetRoot = (Join-Path $PSScriptRoot "..\lossy_tail_peeling_oracle_v7")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Get-Sha256Bytes {
    param([byte[]]$Bytes)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([Convert]::ToHexString($hasher.ComputeHash($Bytes))).ToLowerInvariant()
    }
    finally {
        $hasher.Dispose()
    }
}

function Get-FileRecord {
    param([string]$Path)
    Assert-True (Test-Path -LiteralPath $Path -PathType Leaf) "missing regular file: $Path"
    $item = Get-Item -LiteralPath $Path -Force
    Assert-True (-not (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) "reparse point rejected: $Path"
    $bytes = [IO.File]::ReadAllBytes($item.FullName)
    return [pscustomobject]@{
        Bytes = $bytes
        Length = [int64]$bytes.LongLength
        Sha256 = Get-Sha256Bytes $bytes
    }
}

function Assert-NoDuplicateKeys {
    param(
        [System.Text.Json.JsonElement]$Element,
        [string]$JsonPath
    )
    if ($Element.ValueKind -eq [System.Text.Json.JsonValueKind]::Object) {
        $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($property in $Element.EnumerateObject()) {
            Assert-True ($seen.Add($property.Name)) "duplicate JSON key at $JsonPath.$($property.Name)"
            Assert-NoDuplicateKeys $property.Value "$JsonPath.$($property.Name)"
        }
    }
    elseif ($Element.ValueKind -eq [System.Text.Json.JsonValueKind]::Array) {
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
    $options = [System.Text.Json.JsonDocumentOptions]::new()
    $options.AllowTrailingCommas = $false
    $options.CommentHandling = [System.Text.Json.JsonCommentHandling]::Disallow
    $document = [System.Text.Json.JsonDocument]::Parse($text, $options)
    try {
        Assert-NoDuplicateKeys $document.RootElement '$'
    }
    finally {
        $document.Dispose()
    }
    $value = $text | ConvertFrom-Json -AsHashtable -Depth 100
    return [pscustomobject]@{ Text = $text; Value = $value; Record = $record }
}

function Assert-ZeroSlotSeal {
    param([string]$Path, [string]$Field)
    $parsed = Read-StrictJson $Path
    $claimed = $parsed.Value[$Field]
    Assert-True ($claimed -is [string] -and $claimed -match '^[0-9a-f]{64}$') "invalid seal field $Field in $Path"
    $pattern = '("' + [regex]::Escape($Field) + '"\s*:\s*")([0-9a-f]{64})(")'
    $matcher = [regex]::new($pattern, [Text.RegularExpressions.RegexOptions]::CultureInvariant)
    $matches = $matcher.Matches($parsed.Text)
    Assert-True ($matches.Count -eq 1) "seal slot must occur exactly once in $Path"
    Assert-True ($matches[0].Groups[2].Value -ceq $claimed) "parsed/raw seal mismatch in $Path"
    $zeroed = $matcher.Replace(
        $parsed.Text,
        [Text.RegularExpressions.MatchEvaluator]{
            param($match)
            return $match.Groups[1].Value + ('0' * 64) + $match.Groups[3].Value
        },
        1
    )
    $encoding = [Text.UTF8Encoding]::new($false, $true)
    $actual = Get-Sha256Bytes ($encoding.GetBytes($zeroed))
    Assert-True ($actual -ceq $claimed) "zero-slot internal seal mismatch in $Path"
    return $parsed
}

function Assert-ExactSet {
    param([object[]]$Actual, [string[]]$Expected, [string]$Label)
    $actualSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($entry in $Actual) {
        Assert-True ($entry -is [string]) "$Label contains a non-string"
        Assert-True ($actualSet.Add([string]$entry)) "$Label contains a duplicate: $entry"
    }
    $expectedSet = [System.Collections.Generic.HashSet[string]]::new($Expected, [StringComparer]::Ordinal)
    Assert-True ($actualSet.SetEquals($expectedSet)) "$Label set mismatch"
    Assert-True ($actualSet.Count -eq $Expected.Count) "$Label cardinality mismatch"
}

function Get-Text {
    param([string]$Path)
    $record = Get-FileRecord $Path
    return ([Text.UTF8Encoding]::new($false, $true)).GetString($record.Bytes)
}

function Assert-Contains {
    param([string]$Text, [string]$Needle, [string]$Label)
    Assert-True ($Text.Contains($Needle, [StringComparison]::Ordinal)) "$Label is missing: $Needle"
}

function Assert-InOrder {
    param([string]$Text, [string[]]$Needles, [string]$Label)
    $offset = 0
    foreach ($needle in $Needles) {
        $found = $Text.IndexOf($needle, $offset, [StringComparison]::Ordinal)
        Assert-True ($found -ge 0) "$Label order/member mismatch at: $needle"
        $offset = $found + $needle.Length
    }
}

function Get-SourceBlock {
    param([string]$Text, [string]$Start, [string]$End, [string]$Label)
    $left = $Text.IndexOf($Start, [StringComparison]::Ordinal)
    Assert-True ($left -ge 0) "$Label start missing"
    $right = $Text.IndexOf($End, $left + $Start.Length, [StringComparison]::Ordinal)
    Assert-True ($right -gt $left) "$Label end missing"
    return $Text.Substring($left, $right - $left)
}

function Assert-Rejects {
    param([scriptblock]$Action, [string]$Label)
    $rejected = $false
    try {
        & $Action | Out-Null
    }
    catch {
        $rejected = $true
    }
    Assert-True $rejected "$Label did not reject"
}

function Assert-FiniteTreeModel {
    param([object]$Value)
    if ($null -eq $Value -or $Value -is [string] -or $Value -is [bool]) {
        return
    }
    if ($Value -is [double] -or $Value -is [single] -or $Value -is [decimal] -or
        $Value -is [byte] -or $Value -is [sbyte] -or $Value -is [int16] -or
        $Value -is [uint16] -or $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]) {
        $number = [double]$Value
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
            throw "nonfinite model leaf"
        }
        return
    }
    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($key in $Value.Keys) {
            Assert-FiniteTreeModel $Value[$key]
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        foreach ($entry in $Value) {
            Assert-FiniteTreeModel $entry
        }
        return
    }
    throw "unexpected model leaf"
}

function Invoke-CapabilityTranscriptModel {
    param(
        [object[]]$Records,
        [bool]$WriterClosed,
        [int[]]$PeerCredential,
        [int[]]$ExpectedCredential
    )
    if ($PeerCredential.Count -ne 3 -or $ExpectedCredential.Count -ne 3) { throw "credential arity" }
    for ($index = 0; $index -lt 3; $index++) {
        if ($PeerCredential[$index] -ne $ExpectedCredential[$index]) { throw "peer credential" }
    }
    if ($Records.Count -ne 1) { throw "one record required" }
    if (-not $WriterClosed) { throw "EOF required" }
    if ($null -eq $Records[0]) { throw "empty record" }
    return "CONSUMED_ONCE_BEFORE_THIRD_PARTY_IMPORT"
}

function Select-ReadValidModel {
    param([object[]]$Rows)
    $valid = @($Rows | Where-Object {
        $_.valid -eq $true -and
        [double]$_.maximum_logical -lt 2.0 -and
        [double]$_.maximum_page -lt 2.0
    })
    if ($valid.Count -eq 0) { throw "no read-valid row" }
    return $valid | Sort-Object -Property @{ Expression = { [double]$_.F }; Ascending = $true } | Select-Object -First 1
}

function Get-DecisionStatusModel {
    param(
        [double]$Optimistic,
        [double]$FiniteAbsolute,
        [double]$FiniteCalibrated
    )
    Assert-FiniteTreeModel @($Optimistic, $FiniteAbsolute, $FiniteCalibrated)
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
$manifestPath = Join-Path $auditRoot "audit_manifest.json"
$receiptPath = Join-Path $auditRoot "audit_receipt.json"
$auditManifest = Assert-ZeroSlotSeal $manifestPath "audit_manifest_internal_sha256"
$auditReceipt = Assert-ZeroSlotSeal $receiptPath "audit_receipt_internal_sha256"

Assert-True ($auditManifest.Value["schema"] -ceq "lossy-tail-v7-fresh-independent-source-audit-manifest-v1") "audit manifest schema"
Assert-True ($auditManifest.Value["status"] -ceq "IMMUTABLE_BLOCK_AUDIT_ARTIFACT_SET") "audit manifest status"
Assert-True ($auditReceipt.Value["schema"] -ceq "lossy-tail-v7-fresh-independent-source-audit-receipt-v1") "audit receipt schema"
Assert-True ($auditReceipt.Value["status"] -ceq "BLOCKED_SOURCE_ONLY_RELEASE_CONFORMANCE") "audit receipt status"

$expectedAuditArtifacts = @("README.md", "audit_receipt.json", "verify_audit.ps1")
$artifactRows = @($auditManifest.Value["audit_artifacts"])
Assert-True ($artifactRows.Count -eq $expectedAuditArtifacts.Count) "audit artifact row cardinality"
Assert-ExactSet @($artifactRows | ForEach-Object { $_["path"] }) $expectedAuditArtifacts "audit artifact rows"
foreach ($row in $artifactRows) {
    Assert-ExactSet @($row.Keys) @("path", "bytes", "sha256") "audit artifact row keys"
    $record = Get-FileRecord (Join-Path $auditRoot $row["path"])
    Assert-True ($record.Length -eq [int64]$row["bytes"]) "audit artifact byte mismatch: $($row['path'])"
    Assert-True ($record.Sha256 -ceq [string]$row["sha256"]) "audit artifact hash mismatch: $($row['path'])"
}

$expectedMembers = @(
    "authorization_contract.json", "audit_lock_entrypoint.py", "launch_manifest.json",
    "lossy_tail_core.py", "lossy_tail_oracle.py", "preflight_launch.py",
    "protocol_lock.json", "repair_lock.json", "runtime_calibrate.py",
    "runtime_contract.json", "source_bindings.json"
)
$targetRows = @($auditManifest.Value["target_stage_members"])
Assert-True ($targetRows.Count -eq $expectedMembers.Count) "target stage row cardinality"
Assert-ExactSet @($targetRows | ForEach-Object { $_["path"] }) $expectedMembers "target stage rows"
foreach ($row in $targetRows) {
    Assert-ExactSet @($row.Keys) @("path", "bytes", "sha256") "target stage row keys"
    $record = Get-FileRecord (Join-Path $targetRoot $row["path"])
    Assert-True ($record.Length -eq [int64]$row["bytes"]) "target byte mismatch: $($row['path'])"
    Assert-True ($record.Sha256 -ceq [string]$row["sha256"]) "target hash mismatch: $($row['path'])"
}

$launch = Read-StrictJson (Join-Path $targetRoot "launch_manifest.json")
Assert-True ($launch.Record.Sha256 -ceq "3d5bc5ed95071cc45406d0d2906b54f40d32adad0dffc6323b8fa80ca491ed63") "launch manifest exact SHA-256"
Assert-True ($launch.Value["schema"] -ceq "lossy-tail-v7-launch-manifest-v1") "launch schema"
Assert-ExactSet @($launch.Value["allowed_members"]) $expectedMembers "launch allowed members"
$launchRows = @($launch.Value["members"])
Assert-True ($launchRows.Count -eq 10) "launch member-row cardinality"
Assert-ExactSet @($launchRows | ForEach-Object { $_["path"] }) @($expectedMembers | Where-Object { $_ -cne "launch_manifest.json" }) "launch member rows"
foreach ($row in $launchRows) {
    $audited = @($targetRows | Where-Object { $_["path"] -ceq $row["path"] })
    Assert-True ($audited.Count -eq 1) "launch/audit row join: $($row['path'])"
    Assert-True ([int64]$audited[0]["bytes"] -eq [int64]$row["bytes"]) "launch/audit bytes: $($row['path'])"
    Assert-True ([string]$audited[0]["sha256"] -ceq [string]$row["sha256"]) "launch/audit hash: $($row['path'])"
}

$repair = Read-StrictJson (Join-Path $targetRoot "repair_lock.json")
Assert-True ($repair.Value["repair_lock_sha256"] -ceq "b580de6404add47e7a2b4e27e877a15cf531b41d83f4d690ae2744d0cd67bf56") "repair internal identity"
$identityFiles = [ordered]@{
    scientific_protocol_sha256 = "protocol_lock.json"
    source_bindings_sha256 = "source_bindings.json"
    runtime_contract_sha256 = "runtime_contract.json"
    authorization_contract_sha256 = "authorization_contract.json"
    oracle_bootstrap_sha256 = "lossy_tail_oracle.py"
    scientific_core_sha256 = "lossy_tail_core.py"
    preflight_sha256 = "preflight_launch.py"
    audit_entrypoint_sha256 = "audit_lock_entrypoint.py"
    runtime_calibrate_sha256 = "runtime_calibrate.py"
}
foreach ($label in $identityFiles.Keys) {
    $record = Get-FileRecord (Join-Path $targetRoot $identityFiles[$label])
    Assert-True ([string]$repair.Value["authenticated_identities"][$label] -ceq $record.Sha256) "repair identity mismatch: $label"
}

$bindings = Read-StrictJson (Join-Path $targetRoot "source_bindings.json")
$bindingFiles = @($bindings.Value["files"])
Assert-True ($bindingFiles.Count -eq 12) "source-binding row cardinality"
$expectedExperts = @(24, 24, 56, 56, 80, 80, 88, 88, 96, 96, 120, 120)
$expectedRoles = @("up", "down_transposed", "up", "down_transposed", "up", "down_transposed", "up", "down_transposed", "up", "down_transposed", "up", "down_transposed")
for ($index = 0; $index -lt $bindingFiles.Count; $index++) {
    Assert-True ([int]$bindingFiles[$index]["expert"] -eq $expectedExperts[$index]) "binding expert order"
    Assert-True ([string]$bindingFiles[$index]["role"] -ceq $expectedRoles[$index]) "binding role order"
    Assert-True ([string]$bindingFiles[$index]["name"] -notmatch '[\\/]') "binding basename"
    Assert-True ([string]$bindingFiles[$index]["sha256"] -match '^[0-9a-f]{64}$') "binding hash grammar"
}

$auditEntry = Get-Text (Join-Path $targetRoot "audit_lock_entrypoint.py")
$runtimeCalibrate = Get-Text (Join-Path $targetRoot "runtime_calibrate.py")
$preflight = Get-Text (Join-Path $targetRoot "preflight_launch.py")
$bootstrap = Get-Text (Join-Path $targetRoot "lossy_tail_oracle.py")
$core = Get-Text (Join-Path $targetRoot "lossy_tail_core.py")
$releaseSources = @($auditEntry, $runtimeCalibrate, $preflight, $bootstrap, $core)

foreach ($source in $releaseSources) {
    Assert-True (-not [regex]::IsMatch($source, '(?m)^\s*assert(?:\s|\()')) "release source contains assert"
    Assert-True (-not [regex]::IsMatch($source, '(?m)^\s*(?:from|import)\s+(?:requests|urllib|http|httpx|ftplib|paramiko|transformers|torch)\b')) "forbidden network/model import"
    Assert-True (-not $source.Contains('.connect(', [StringComparison]::Ordinal)) "external socket connect surface"
    Assert-True (-not $source.Contains('create_connection(', [StringComparison]::Ordinal)) "external socket connection surface"
}
foreach ($entry in @($auditEntry, $runtimeCalibrate, $preflight, $bootstrap)) {
    Assert-Contains $entry 'sys.flags.optimize != 0' "optimization-mode firewall"
}
Assert-Contains $core 'sys.flags.optimize != 0' "scientific-core optimization firewall"
Assert-True ($core.IndexOf('_v7_preimport_production_firewall(_V7_CORE_CONTEXT)', [StringComparison]::Ordinal) -lt $core.IndexOf('import numpy as np', [StringComparison]::Ordinal)) "core firewall must precede NumPy"
Assert-True (-not [regex]::IsMatch($auditEntry, '(?m)^\s*(?:from|import)\s+(?:numpy|cupy)\b')) "audit entrypoint third-party import surface"

Assert-InOrder $preflight @(
    'socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)',
    'subprocess.Popen(',
    'parent_channel.send(canonical_bytes(capability))',
    'parent_channel.shutdown(socket.SHUT_WR)',
    'acknowledgement_payload = parent_channel.recv(65536)',
    'if parent_channel.recv(1) != b""',
    'child_exit = process.wait()'
) "preflight one-use transcript"
Assert-InOrder $bootstrap @(
    'os.set_inheritable(descriptor, False)',
    'channel = socket.socket(fileno=descriptor)',
    'socket.SO_PEERCRED',
    'peer_pid, peer_uid, peer_gid = struct.unpack("3i", peer_payload)',
    'if peer_pid != os.getppid() or peer_uid != os.getuid() or peer_gid != os.getgid()',
    'message = channel.recv(65536)',
    'if channel.recv(1) != b""',
    'live_cmdline = read_regular(Path(f"/proc/{peer_pid}/cmdline")',
    'channel.send(canonical(acknowledgement))',
    'channel.close()',
    'exec(compile(payloads["lossy_tail_core.py"]'
) "child peer/EOF/descriptor-exec transcript"

Assert-True ((Invoke-CapabilityTranscriptModel @([pscustomobject]@{ nonce = "n" }) $true @(101, 1000, 1000) @(101, 1000, 1000)) -ceq "CONSUMED_ONCE_BEFORE_THIRD_PARTY_IMPORT") "capability transcript acceptance"
Assert-Rejects { Invoke-CapabilityTranscriptModel @() $true @(101, 1000, 1000) @(101, 1000, 1000) } "missing capability record"
Assert-Rejects { Invoke-CapabilityTranscriptModel @("one", "two") $true @(101, 1000, 1000) @(101, 1000, 1000) } "extra capability record"
Assert-Rejects { Invoke-CapabilityTranscriptModel @("one") $false @(101, 1000, 1000) @(101, 1000, 1000) } "missing capability EOF"
Assert-Rejects { Invoke-CapabilityTranscriptModel @("one") $true @(999, 1000, 1000) @(101, 1000, 1000) } "wrong peer PID"

$readRows = @(
    [pscustomobject]@{ valid = $true; F = 0.1; maximum_logical = 2.25; maximum_page = 1.0 },
    [pscustomobject]@{ valid = $true; F = 0.2; maximum_logical = 1.25; maximum_page = 1.25 }
)
$readWinner = Select-ReadValidModel $readRows
Assert-True ([Math]::Abs([double]$readWinner.F - 0.2) -lt 1.0e-15) "read-valid adversary selected global invalid row"
Assert-InOrder $core @(
    'best_uniform_global = best_scored(uniform, "raw uniform global", require_read_valid=False)',
    'best_uniform = best_scored(uniform, "raw uniform read-valid", require_read_valid=True)',
    'winner = best_scored(',
    'require_read_valid=True',
    'best_xklt_global = best_scored(uniform_xklt_rows, "support-XKLT global", require_read_valid=False)',
    'best_xklt = best_scored(uniform_xklt_rows, "support-XKLT read-valid", require_read_valid=True)'
) "read-valid selection flow"
Assert-Contains $core 'maximum_logical < 2.0 and maximum_page < 2.0' "strict read maximum rule"
Assert-Contains $core 'expert["cold_logical_amplification"], "expert logical") < 2.0' "strict expert logical rule"
Assert-Contains $core 'expert["cold_page_amplification"], "expert page") < 2.0' "strict expert page rule"

Assert-FiniteTreeModel @{ outer = @(@{ inner = 1.0 }, -2.0) }
Assert-Rejects { Assert-FiniteTreeModel @{ nested = [double]::NaN } } "nested NaN"
Assert-Rejects { Assert-FiniteTreeModel @{ nested = [double]::PositiveInfinity } } "nested positive infinity"
Assert-Rejects { Assert-FiniteTreeModel @{ nested = [double]::NegativeInfinity } } "nested negative infinity"
Assert-Contains $core 'require_finite_tree(row, f"decision row[{ordinal}]")' "recursive decision finiteness"
Assert-Contains $core 'control_std = finite_scalar(np.std(control_s, ddof=1)' "control standard-deviation finiteness"
Assert-Contains $core 'finite_joint > optimistic_m + DECISION_CONSISTENCY_EPSILON_S' "finite/optimistic consistency"

$targetS = -0.5 * [Math]::Log(0.8, 2.0)
$killS = $targetS - 0.02
Assert-True ((Get-DecisionStatusModel 0.14 0.1 0.1) -ceq "EARLY_KILL_FAR_SHORT") "early-kill model"
Assert-True ((Get-DecisionStatusModel $killS 0.1 0.1) -ceq "HOLD_NUMERIC_BOUNDARY") "kill-boundary model"
Assert-True ((Get-DecisionStatusModel 0.15 0.13 0.13) -ceq "HOLD_OPTIMISTIC_NEAR_BOUNDARY") "near-boundary hold model"
Assert-True ((Get-DecisionStatusModel 0.18 0.17 0.17) -ceq "FINITE_CODEC_WARRANTED") "finite promotion model"
Assert-True ((Get-DecisionStatusModel 0.18 0.13 0.13) -ceq "OPTIMISTIC_SURVIVOR") "optimistic survivor model"
Assert-Rejects { Get-DecisionStatusModel ([double]::NaN) 0.2 0.2 } "decision NaN"
Assert-Rejects { Get-DecisionStatusModel ([double]::PositiveInfinity) 0.2 0.2 } "decision positive infinity"
Assert-Rejects { Get-DecisionStatusModel ([double]::NegativeInfinity) 0.2 0.2 } "decision negative infinity"
Assert-Contains $core 'mse = total_sse / source_energy' "source-relative MSE"
Assert-Contains $core 'actual_rate = physical_bits / PANEL_N' "physical rate"
Assert-Contains $core 'f_value = mse * 2.0 ** (2.0 * actual_rate)' "F definition"
Assert-Contains $core 'optimistic_m < KILL_THRESHOLD_S' "strict early-kill threshold"

$runtimeContract = Read-StrictJson (Join-Path $targetRoot "runtime_contract.json")
$requiredMemoryKeys = @($runtimeContract.Value["probe"]["memory_evidence_per_cell"])
Assert-ExactSet $requiredMemoryKeys @(
    "stream_synchronized", "used_bytes_before_free", "total_bytes_before_free",
    "used_bytes_after_free", "total_bytes_after_free",
    "all_per_cell_gpu_arrays_deleted_before_free"
) "runtime memory evidence contract"
$stableBlock = Get-SourceBlock $core 'for ordinal, host in enumerate(adversaries):' 'core = {"runtime_tuple": runtime' "stable-order memory block"
Assert-Contains $stableBlock 'used_bytes_before_free' "stable-order used-before evidence"
Assert-Contains $stableBlock 'used_bytes_after_free' "stable-order used-after evidence"
Assert-Contains $stableBlock 'total_bytes_after_free' "stable-order total-after evidence"
Assert-True (-not $stableBlock.Contains('total_bytes_before_free', [StringComparison]::Ordinal)) "audited blocker disappeared: stable-order total-before is now recorded"
Assert-True (-not $stableBlock.Contains('all_per_cell_gpu_arrays_deleted_before_free', [StringComparison]::Ordinal)) "audited blocker disappeared: stable-order deletion assertion is now recorded"

$writeBlock = Get-SourceBlock $core 'def write_sealed_json(' 'def bf16_words_to_float32(' "production output writer"
Assert-InOrder $writeBlock @('os.mkdir(path.parent', 'os.open(os.fspath(path)') "output path creation/open"
Assert-True (-not $writeBlock.Contains('dir_fd=', [StringComparison]::Ordinal)) "audited blocker disappeared: output open is now directory-descriptor-bound"
Assert-True (-not $writeBlock.Contains('os.fstat(', [StringComparison]::Ordinal)) "audited blocker disappeared: output parent is now reauthenticated"

$panelBlock = Get-SourceBlock $core 'def build_panel(' 'def raw_components(' "panel cleanup"
Assert-InOrder $panelBlock @('del pair_x, pair_words, pair_masks', 'cp.get_default_memory_pool().free_all_blocks()') "panel pool release"
Assert-True (-not $panelBlock.Contains('del masks', [StringComparison]::Ordinal)) "audited blocker disappeared: retained mask loop local is now deleted"

Assert-Contains $preflight 'if value.get("status") != section["required_status"]' "preflight audit-status comparison"
Assert-Contains $core 'if value.get("status") != section["required_status"]' "core audit-status comparison"
$authorizationContract = Read-StrictJson (Join-Path $targetRoot "authorization_contract.json")
Assert-True (-not $authorizationContract.Value["required_values"].Contains("source_audit")) "audited blocker disappeared: source-audit PASS value is now fixed"
Assert-True (-not $authorizationContract.Value["required_values"].Contains("runtime_audit")) "audited blocker disappeared: runtime-audit PASS value is now fixed"

Assert-Contains $core '/proc/{context[''parent_pid'']}/cmdline' "live-parent command-line proof"
Assert-Contains $core '/proc/{context[''parent_pid'']}/exe' "live-parent executable proof"
Assert-Contains $bootstrap 'preflight parent cmdline' "bootstrap parent command-line proof"

$blockers = @($auditReceipt.Value["blockers"])
$expectedBlockerIds = @(
    "PARENT_PROVENANCE_RESTS_ON_SELF_MUTABLE_CMDLINE",
    "AUDIT_REQUIRED_STATUS_IS_AUTHORIZATION_CHOSEN_NOT_PASS_PINNED",
    "OUTPUT_CREATE_OPEN_IS_PATH_BASED_AND_RACEABLE",
    "STABLE_ORDER_MEMORY_EVIDENCE_VIOLATES_RUNTIME_CONTRACT",
    "PANEL_POOL_FREE_RUNS_WITH_LIVE_LOOP_LOCALS"
)
Assert-True ($blockers.Count -eq $expectedBlockerIds.Count) "receipt blocker cardinality"
Assert-ExactSet @($blockers | ForEach-Object { $_["id"] }) $expectedBlockerIds "receipt blockers"
$ledger = $auditReceipt.Value["access_ledger"]
foreach ($key in @(
    "model_or_qwen_paths_traversed", "model_payload_files_opened", "cupy_imports",
    "cuda_initializations", "gpu_jobs", "network_calls", "producer_tests_executed",
    "producer_receipts_trusted", "runtime_calibrations", "production_runs",
    "production_authorizations_created"
)) {
    Assert-True ([int64]$ledger[$key] -eq 0) "nonzero audit access ledger: $key"
}
Assert-True ($auditReceipt.Value["authorization"] -ceq "NONE; BLOCK RECEIPT MUST NEVER BE USED AS LAUNCH AUTHORITY") "receipt authorization boundary"

$result = [ordered]@{
    verifier = "lossy-tail-v7-fresh-independent-source-audit-verifier-v1"
    audited_launch_manifest_sha256 = $launch.Record.Sha256
    target_stage_member_count = $targetRows.Count
    independent_probe_groups = 17
    blocker_count = $blockers.Count
    status = "BLOCKED_SOURCE_ONLY_RELEASE_CONFORMANCE"
    payload_authorized = $false
    model_payload_files_opened = 0
    cupy_imports = 0
    gpu_jobs = 0
    network_calls = 0
}
Write-Output ($result | ConvertTo-Json -Compress)
