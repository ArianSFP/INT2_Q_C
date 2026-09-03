param(
    [Parameter(Mandatory = $true)] [string]$Producer,
    [Parameter(Mandatory = $true)] [string]$Output
)

$ErrorActionPreference = 'Stop'
$ExpectedManifest = '7901e78eaf7c6b854d7bfaa2afbb4eb7be337449a72ef66e66d00adb87f64ab4'
$ExpectedRoot = '14ec1fdc19435f4f3655b4f3458ef774a6503d9c88c2d62c510815499c14aecd'
$Disposition = 'PASS_EXACT_SOURCE_AND_EFFECTIVE_COMPILED_HOLD__BLOCK_LITERAL_REPLAY_SCORING_COUNT_PACKET_ALIAS_AND_TRACE_CLAIMS__HOLD_PAYLOAD_AND_RD'

function Get-LowerSha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [Convert]::ToHexString($sha.ComputeHash($Bytes)).ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "BMP/QTT6 v3 independent review failed: $Message" }
}

function Json-String([string]$Value) {
    return ($Value | ConvertTo-Json -Compress)
}

function Source-Section([string]$Source, [string]$Start, [string]$End) {
    $left = $Source.IndexOf($Start, [StringComparison]::Ordinal)
    Require ($left -ge 0) "section start $Start"
    $right = $Source.IndexOf($End, $left + $Start.Length, [StringComparison]::Ordinal)
    Require ($right -gt $left) "section end $End"
    return $Source.Substring($left, $right - $left)
}

# Reject the caller's root object before resolving it.
$original = Get-Item -LiteralPath $Producer -Force
Require ($original.PSIsContainer) 'producer directory'
Require (($original.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'producer root is not a link'
$root = (Resolve-Path -LiteralPath $Producer).Path
$manifestPath = Join-Path $root 'SOURCE_MANIFEST.json'
$manifestBytes = [IO.File]::ReadAllBytes($manifestPath)
$manifestHash = Get-LowerSha256 $manifestBytes
Require ($manifestHash -eq $ExpectedManifest) 'producer manifest external pin'
$manifest = [Text.Json.JsonSerializer]::Deserialize[Text.Json.JsonElement]($manifestBytes)
Require ($manifest.GetProperty('schema').GetString() -eq 'strata-bmp-obdd-qtt6-v3-authority-source-manifest-v1') 'producer schema'
Require ($manifest.GetProperty('source_root_sha256').GetString() -eq $ExpectedRoot) 'producer root pin'

$names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$canonicalRows = @()
foreach ($row in $manifest.GetProperty('members').EnumerateArray()) {
    $name = $row.GetProperty('name').GetString()
    Require ($names.Add($name)) "unique member $name"
    Require ([IO.Path]::GetFileName($name) -ceq $name) "flat member $name"
    $item = Get-Item -LiteralPath (Join-Path $root $name) -Force
    Require (-not $item.PSIsContainer) "regular member $name"
    Require (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "non-link member $name"
    $bytes = [IO.File]::ReadAllBytes($item.FullName)
    $length = $row.GetProperty('bytes').GetInt64()
    $hash = $row.GetProperty('sha256').GetString()
    Require ($bytes.LongLength -eq $length) "member length $name"
    Require ((Get-LowerSha256 $bytes) -eq $hash) "member hash $name"
    $canonicalRows += '{"bytes":' + $length + ',"name":' + (Json-String $name) + ',"sha256":' + (Json-String $hash) + '}'
}
Require ($canonicalRows.Count -eq 8) 'eight producer members'
$actual = @((Get-ChildItem -LiteralPath $root -Force).Name | Sort-Object) -join "`n"
$wanted = @(@($names) + 'SOURCE_MANIFEST.json' | Sort-Object) -join "`n"
Require ($actual -ceq $wanted) 'producer exact flat closure'
$rootHash = Get-LowerSha256 ([Text.Encoding]::ASCII.GetBytes('[' + ($canonicalRows -join ',') + ']'))
Require ($rootHash -eq $ExpectedRoot) 'producer independently recomputed source root'

$authority = [IO.File]::ReadAllText((Join-Path $root 'authority.py'))
$tests = [IO.File]::ReadAllText((Join-Path $root 'test_source_only.py'))
$commonExecution = Source-Section $authority 'def _validate_common_execution' 'def _validate_common_audit'
$adapter = Source-Section $authority 'def _validate_adapter_details' 'def _validate_scorer_details'
$scorer = Source-Section $authority 'def _validate_scorer_details' 'def _validate_read_details'
$reader = Source-Section $authority 'def _validate_read_details' 'def _validate_launch_audit_details'
$sourceRecords = Source-Section $authority 'def _source_records' 'def _bind_capabilities_to_routes'
$binding = Source-Section $authority 'def _bind_capabilities_to_routes' 'def verify_precommitted_evidence'
$verification = Source-Section $authority 'def verify_precommitted_evidence' 'def authorize_production'

$verified = [ordered]@{
    compiled_launch_pin_is_absent = $authority.Contains('TRUSTED_LAUNCH_MANIFEST_SHA256: str | None = None')
    production_entry_fails_before_payload_path = ($authority.IndexOf('HOLD: no independently frozen production launch-manifest pin') -lt $authority.IndexOf('result = verify_precommitted_evidence('))
    capability_manifest_root_execution_and_audit_hashes_pinned = ($authority.Contains('execution_receipt_sha256') -and $authority.Contains('audit_receipt_sha256') -and $authority.Contains('capability canonical source root'))
    capability_exact_flat_closure = $authority.Contains('capability exact regular closure')
    fixture_dummy_and_self_authored_true_flags_rejected = $authority.Contains('production capability cannot be fixture, dummy, or self-authored')
    producer_executor_auditor_strings_distinct_within_capability = $authority.Contains('producer, executor, and auditor must be independent')
    literal_packet_hash_and_size_reopened = $authority.Contains('literal STRATA packet bytes')
    source_path_inode_and_content_aliases_rejected = ($authority.Contains('source inode alias across routes/roles') -and $authority.Contains('source-byte alias across model/control routes') -and $authority.Contains('source-path alias across routes/roles'))
    adapter_requires_scale_transform_framing_and_canonical_fields = ($adapter.Contains('scale_payload_inside_packet') -and $adapter.Contains('forward_transform_sha256') -and $adapter.Contains('framing_header_bytes') -and $adapter.Contains('canonical_reencode_equal'))
    scorer_formulae_recomputed_from_receipt_numbers = ($scorer.Contains('sse_fp64') -and $scorer.Contains('source_energy_fp64') -and $scorer.Contains('f_value = relative * 2.0 ** (2.0 * rate)'))
    scorer_authority_strings_differ_from_adapter = $binding.Contains('BF16 scorer independent of STRATA adapter')
    event_offsets_sizes_and_page_hashes_checked = ($reader.Contains('event["file_offset"] == event["page_index"] * PAGE_BYTES') -and $binding.Contains('read page binds literal packet'))
    read_amplification_recomputed_per_route = ($reader.Contains('amplification = physical / row["literal_packet_bytes"]') -and $reader.Contains('amplification < 2.0'))
    model_source_files_are_even_length = $sourceRecords.Contains('len(payload) % 2 == 0')
}
foreach ($item in $verified.GetEnumerator()) { Require ([bool]$item.Value) "verified mechanism $($item.Key)" }

$gaps = [ordered]@{
    lower_level_verifier_can_label_caller_pinned_evidence_production = $verification.Contains('"production_authorized": not allow_source_test_fixture')
    launch_v3_manifest_and_root_are_only_well_formed_not_equal_to_producer = ($verification.Contains('is_sha256(record["v3_source_manifest_sha256"])') -and -not $authority.Contains($ExpectedManifest) -and -not $authority.Contains($ExpectedRoot))
    execution_output_digest_has_no_literal_output_artifact_binding = ($commonExecution.Contains('"output_sha256"') -and -not $commonExecution.Contains('regular_bytes('))
    independence_is_nominal_ids_without_cross_capability_separation = ($authority.Contains('len({producer, executor, auditor}) == 3') -and -not $verification.Contains('auditor_authority_id'))
    scale_membership_and_canonical_reencode_are_boolean_attestations = ($adapter.Contains('row["scale_payload_inside_packet"] is True') -and $adapter.Contains('row["canonical_reencode_equal"] is True') -and -not $adapter.Contains('regular_bytes('))
    transform_hashes_are_well_formed_but_not_pinned_to_current_strata_code = ($adapter.Contains('forward_transform_sha256') -and -not $adapter.Contains('implementation_sha256'))
    scorer_does_not_decode_literal_bf16_or_open_reconstruction = ($scorer.Contains('float(row["sse_fp64"])') -and -not $scorer.Contains('regular_bytes(') -and -not $scorer.Contains('frombytes'))
    source_adapter_scorer_weight_counts_are_not_cross_bound = (-not $binding.Contains('decoded_weight_count') -and -not $sourceRecords.Contains('weight_count'))
    packet_path_inode_and_content_aliases_are_not_rejected = ($sourceRecords.Contains('packet_payload') -and -not $sourceRecords.Contains('packet inode alias') -and -not $sourceRecords.Contains('packet-byte alias'))
    trace_may_omit_packet_pages = ($binding.Contains('for event in read["events"]') -and -not $binding.Contains('set(range(page_count))'))
    launch_path_symlink_is_resolved_before_regular_file_check = ($verification.Contains('launch = launch_manifest_path.resolve(strict=True)') -and -not $verification.Contains('launch_manifest_path.is_symlink'))
    no_qwen_minus_control_or_per_family_acceptance = ($verification.Contains('model_f <= TARGET_F') -and -not $verification.Contains('source_specific') -and -not $verification.Contains('strongest_control'))
}
foreach ($item in $gaps.GetEnumerator()) { Require ([bool]$item.Value) "gap observation $($item.Key)" }

$coverage = [ordered]@{
    producer_tests_count_claimed = 14
    producer_tests_cover_compiled_hold = $tests.Contains('test_02_production_is_compiled_hold')
    producer_tests_cover_true_self_authored_flag = $tests.Contains('test_07_self_authored_receipt_rejected')
    producer_tests_cover_scale_zero_only = $tests.Contains('test_08_adapter_requires_literal_scale_bytes')
    producer_tests_cover_formula_tamper_only = $tests.Contains('test_09_bf16_scorer_arithmetic_recomputed')
    producer_tests_cover_layout_flag_and_ratio = ($tests.Contains('test_10_layout_is_not_read_trace') -and $tests.Contains('test_11_read_amplification_is_recomputed'))
    producer_tests_do_not_cover_packet_alias = -not $tests.Contains('packet_alias')
    producer_tests_do_not_cover_incomplete_page_trace = -not $tests.Contains('omit')
}
foreach ($item in $coverage.GetEnumerator()) {
    if ($item.Key -ne 'producer_tests_count_claimed') { Require ([bool]$item.Value) "coverage observation $($item.Key)" }
}

$receipt = [ordered]@{
    schema = 'strata-bmp-qtt6-gate-v3-authority-independent-static-review-receipt-v1'
    producer_manifest_sha256 = $manifestHash
    producer_source_root_sha256 = $rootHash
    producer_members = $canonicalRows.Count
    exact_flat_closure = $true
    verified_mechanisms = $verified
    blocking_gap_observations = $gaps
    producer_test_coverage = $coverage
    static_checks = $verified.Count + $gaps.Count + ($coverage.Count - 1)
    model_qwen_strata_control_or_packet_payload_accessed = $false
    network_used = $false
    producer_modified = $false
    python_tests_executed = $false
    runtime_executed = $false
    disposition = $Disposition
}
$json = $receipt | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText($Output, $json + "`n", [Text.UTF8Encoding]::new($false))
$json
