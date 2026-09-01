[CmdletBinding()]
param(
    [string]$Producer = (Join-Path $PSScriptRoot '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'AUDIT FAILED: pwsh 7 or newer is required' }

$script:Checks = [int64]0
function Check([bool]$Condition,[string]$Message) {
    if (-not $Condition) { throw "AUDIT FAILED: $Message" }
    $script:Checks++
}
function Equal($Actual,$Expected,[string]$Message) {
    if ($Actual -ne $Expected) { throw "AUDIT FAILED: $Message; actual=$Actual expected=$Expected" }
    $script:Checks++
}
function Close([double]$Actual,[double]$Expected,[double]$Tolerance,[string]$Message) {
    if ([math]::Abs($Actual-$Expected) -gt $Tolerance) {
        throw "AUDIT FAILED: $Message; actual=$Actual expected=$Expected"
    }
    $script:Checks++
}
function Sha-Bytes([byte[]]$Bytes) {
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}
function Sha-Text([string]$Text) { Sha-Bytes ([Text.Encoding]::UTF8.GetBytes($Text)) }
function File-Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Read-Utf8([string]$Path) { [IO.File]::ReadAllText($Path,[Text.UTF8Encoding]::new($false,$true)) }
function Contains-Token([string]$Text,[string]$Token,[string]$Message) {
    Check ($Text.Contains($Token,[StringComparison]::Ordinal)) $Message
}
function Before-Token([string]$Text,[string]$Left,[string]$Right,[string]$Message) {
    $leftIndex=$Text.IndexOf($Left,[StringComparison]::Ordinal)
    $rightIndex=$Text.IndexOf($Right,[StringComparison]::Ordinal)
    Check ($leftIndex -ge 0 -and $rightIndex -ge 0 -and $leftIndex -lt $rightIndex) $Message
}
function Line-Of([string]$Text,[string]$Token) {
    $index=$Text.IndexOf($Token,[StringComparison]::Ordinal)
    if($index -lt 0){return 0}
    return 1+([regex]::Matches($Text.Substring(0,$index),"`n").Count)
}
function Is-Regular-NoLink([IO.FileSystemInfo]$Item) {
    return (-not $Item.PSIsContainer) -and (-not ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint))
}

function Parse-Manifest([byte[]]$Raw,[string[]]$ExpectedNames) {
    $text=[Text.Encoding]::ASCII.GetString($Raw)
    if($text.Contains("`r",[StringComparison]::Ordinal)){throw 'manifest CR is forbidden'}
    if(-not $text.EndsWith("`n",[StringComparison]::Ordinal)){throw 'manifest terminal LF missing'}
    $body=$text.Substring(0,$text.Length-1)
    $lines=if($body.Length){$body.Split("`n",[StringSplitOptions]::None)}else{@()}
    $rows=[ordered]@{}
    $previous=$null
    foreach($line in $lines){
        if($line -cnotmatch '^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$'){throw 'manifest row malformed'}
        $hash=$Matches[1]; $name=$Matches[2]
        if($rows.Contains($name)){throw 'manifest duplicate member'}
        if($null -ne $previous -and [StringComparer]::Ordinal.Compare($name,$previous) -le 0){throw 'manifest not strictly sorted'}
        $rows[$name]=$hash; $previous=$name
    }
    $actual=@($rows.Keys); $want=@($ExpectedNames)
    [Array]::Sort($actual,[StringComparer]::Ordinal); [Array]::Sort($want,[StringComparer]::Ordinal)
    if(($actual -join "`n") -cne ($want -join "`n")){throw 'manifest member set mismatch'}
    return $rows
}
function Rejects-Manifest([byte[]]$Raw,[string[]]$ExpectedNames) {
    try { $null=Parse-Manifest $Raw $ExpectedNames; return $false } catch { return $true }
}

function Path-AncestorOrEqual([string]$Left,[string]$Right) {
    $leftNorm=$Left.TrimEnd('/')
    return $Right -ceq $leftNorm -or $Right.StartsWith($leftNorm+'/',[StringComparison]::Ordinal)
}
function Synthetic-Pair-Rejects([hashtable]$Left,[hashtable]$Right) {
    if((Path-AncestorOrEqual $Left.path $Right.path) -or (Path-AncestorOrEqual $Right.path $Left.path)){return $true}
    if($Left.exists -and $Right.exists -and $Left.device -eq $Right.device -and $Left.inode -eq $Right.inode){return $true}
    if($Left.major_minor -eq $Right.major_minor -and (
        (Path-AncestorOrEqual $Left.filesystem_path $Right.filesystem_path) -or
        (Path-AncestorOrEqual $Right.filesystem_path $Left.filesystem_path))){return $true}
    return $false
}

$producerRoot=[IO.Path]::GetFullPath($Producer)
Check (Test-Path -LiteralPath $producerRoot -PathType Container) 'producer directory exists'
$manifestPath=Join-Path $producerRoot 'ARTIFACT_SHA256SUMS.txt'
Equal (File-Sha $manifestPath) 'e9ef19853d1350ac4085a18a65b19a6805620ef7a091447ce57404715f88805f' 'frozen producer manifest'
$expectedMembers=@(
    'README.md','candidate_lock.json','common.py','kernels.py','overlay.py','parity.py','source_trace.py',
    'test_bootstrap.py','test_common.py','test_overlay.py','test_parity.py','test_source_trace.py',
    'test_tier_c_gate.py','tier_c_gate.py','verify_prelaunch.py','verify_v4_reuse.py'
)
$manifestRaw=[IO.File]::ReadAllBytes($manifestPath)
$manifestRows=Parse-Manifest $manifestRaw $expectedMembers
Equal $manifestRows.Count 16 'manifest row count'
$actualMembers=@(Get-ChildItem -LiteralPath $producerRoot -Force)
Equal $actualMembers.Count 17 'exact closure count'
foreach($item in $actualMembers){
    Check (Is-Regular-NoLink $item) "regular non-link closure member $($item.Name)"
    Check ($item.Name -ceq 'ARTIFACT_SHA256SUMS.txt' -or $manifestRows.Contains($item.Name)) "expected closure member $($item.Name)"
}
foreach($entry in $manifestRows.GetEnumerator()){
    $path=Join-Path $producerRoot $entry.Key
    Check (Test-Path -LiteralPath $path -PathType Leaf) "manifested member exists $($entry.Key)"
    Equal (File-Sha $path) $entry.Value "manifested member hash $($entry.Key)"
}

$candidatePath=Join-Path $producerRoot 'candidate_lock.json'
$candidateRaw=[IO.File]::ReadAllBytes($candidatePath)
$candidateText=[Text.Encoding]::UTF8.GetString($candidateRaw)
Equal (Sha-Bytes $candidateRaw) 'cec0b12927340d82c1c8c78cd02b7849c7371b1b16d7f2c1b1d24ae889ada58d' 'candidate lock file hash'
$sealMatches=[regex]::Matches($candidateText,'("lock_sha256"\s*:\s*")([0-9a-f]{64})(")')
Equal $sealMatches.Count 1 'one internal lock seal'
$seal=$sealMatches[0]
$normalized=$candidateText.Substring(0,$seal.Groups[2].Index)+'TO_BE_FILLED_AFTER_CANONICAL_FREEZE'+$candidateText.Substring($seal.Groups[2].Index+$seal.Groups[2].Length)
$internalSeal=Sha-Text $normalized
Equal $internalSeal '4b6b3a73e4fca1175adb26b492978a3e42bcf2ce8af4296d45366b104b568a6e' 'candidate lock internal seal'
$lock=$candidateText|ConvertFrom-Json -Depth 100
Equal ([string]$lock.lock_sha256) $internalSeal 'internal seal literal'
Equal ([string]$lock.schema) 'qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_candidate_lock_v5' 'lock schema'
Equal ([string]$lock.status) 'FROZEN_SOURCE_ONLY_V5_RAW_ENTRYPOINT_AND_OUTPUT_BOUNDARY_NOT_GPU_OR_PAYLOAD_AUTHORIZATION' 'lock status'
Equal ([bool]$lock.sealed) $true 'lock sealed'
Equal ([bool]$lock.sealed_protocol.runtime_or_gpu_authorized) $false 'runtime not authorized'
Equal ([bool]$lock.sealed_protocol.qwen_payload_or_manifest_authorized) $false 'payload not authorized'

$dependencies=[ordered]@{
    '..\initialization_anchor_layout_expansion_v1\ARTIFACT_SHA256SUMS.txt'='cda2a770e03110e3a8c9a31af2fc1b16fa0753836241e8af4c4398039e2e3244'
    '..\initialization_anchor_layout_expansion_v1_design_audit\ARTIFACT_SHA256SUMS.txt'='8455d86d8143f8b729a37041aca6e37637e4c73d3e29cb970d7e00342ddbd174'
    '..\tier_c_grouped_v5_layout_overlay_v1_source_audit\ARTIFACT_SHA256SUMS.txt'='16b41a79e663440cff1db6a4b53408160069d2270b565a4aee3da1c19f01af7b'
    '..\tier_c_grouped_v5_layout_overlay_v2_source_audit\ARTIFACT_SHA256SUMS.txt'='23943f35887e321b285437a8ca517f59bc749a7637500ff1b6bb89af8b8f3705'
    '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v3\ARTIFACT_SHA256SUMS.txt'='0b0a69cc209037cd9130d6b5bba8b9e920c9b398f32ba2fb0862a1cd4b3a292d'
    '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v3_source_audit\ARTIFACT_SHA256SUMS.txt'='0c23f0afc98611de8ae36b32c4a9959fe1cb7c16142fcdf44fe131fb529351dd'
    '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v3_source_audit\audit_receipt.json'='64ec0ab258c916259b7d7b4ce73be6929385c8e68bbc86ad1c25ca0c8c131844'
    '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v4\ARTIFACT_SHA256SUMS.txt'='dbcc8ce2c7bc63c90fa36f01e6353a72f5c2572170a4a98ad607c11481445f97'
    '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v4_source_audit\ARTIFACT_SHA256SUMS.txt'='095d94ff55677a4c5542f3c3e711d49952a64df788eb9812fe216a82db0f0d87'
    '..\initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v4_source_audit\audit_receipt.json'='42be2a15a8ab5ed1383a76c1f2f41a634052a26a0c525066c00f2a456f65e846'
    '..\initialization_anchor_oracle_tier_c_grouped_v4\ARTIFACT_SHA256SUMS.txt'='19f5b729230f90413a6e1f8c1ef2b0421c94f14635abf57f9b2d7f000599f715'
    '..\tier_c_grouped_v4_source_audit\ARTIFACT_SHA256SUMS.txt'='fc2f7ad554436e2136c224bd46a5b0c3beee62d69e51c352304b8943da0db028'
    '..\tier_c_grouped_v4_calibration_audit\ARTIFACT_SHA256SUMS.txt'='0c7ca1241bad8bd3b99c807c50945b371248a6b77820bc72244b5d433604a621'
    '..\tier_c_grouped_v4_result_audit\ARTIFACT_SHA256SUMS.txt'='ae4e1f38b5602e2c43355e4c68604bca65dbb974724b328ca78e10367e9b992e'
    '..\legacy_packed_descriptor_execution_v1\CODE_MANIFEST.sha256'='2468baeb6d962b3e8a305b87791fdc4663b29c0f79beaa5957411653ade1c44f'
    '..\legacy_packed_descriptor_execution_v1\result_2468baeb\ARTIFACT_MANIFEST.sha256'='b44052815d0f3be4e437d56e60cdca4fcdc941fd750bc40fa535363d37080be2'
}
foreach($entry in $dependencies.GetEnumerator()){
    $path=[IO.Path]::GetFullPath((Join-Path $producerRoot $entry.Key))
    Check (Test-Path -LiteralPath $path -PathType Leaf) "frozen dependency exists $($entry.Key)"
    $item=Get-Item -LiteralPath $path -Force
    Check (Is-Regular-NoLink $item) "frozen dependency regular non-link $($entry.Key)"
    Equal (File-Sha $path) $entry.Value "frozen dependency hash $($entry.Key)"
}

$bootstrap=Read-Utf8 (Join-Path $producerRoot 'verify_prelaunch.py')
$common=Read-Utf8 (Join-Path $producerRoot 'common.py')
$runner=Read-Utf8 (Join-Path $producerRoot 'tier_c_gate.py')
$trace=Read-Utf8 (Join-Path $producerRoot 'source_trace.py')
$reuse=Read-Utf8 (Join-Path $producerRoot 'verify_v4_reuse.py')
$testBootstrap=Read-Utf8 (Join-Path $producerRoot 'test_bootstrap.py')
$testCommon=Read-Utf8 (Join-Path $producerRoot 'test_common.py')
$testTrace=Read-Utf8 (Join-Path $producerRoot 'test_source_trace.py')
$testRunner=Read-Utf8 (Join-Path $producerRoot 'test_tier_c_gate.py')

# Recheck every v4 blocker repair at the source level.
Before-Token $bootstrap 'raw = sys.argv[0]' 'PACKAGE = Path(str(RAW_ENTRYPOINT_IDENTITY["raw_argv0"])).parent' 'raw argv0 captured before package derivation'
Contains-Token $bootstrap 'if not os.path.isabs(raw):' 'raw argv0 absolute check'
Contains-Token $bootstrap 'if raw != os.path.normpath(raw)' 'raw argv0 canonical lexical check'
Contains-Token $bootstrap 'if raw != EXPECTED_CANONICAL_ENTRYPOINT:' 'raw argv0 exact path check'
Contains-Token $bootstrap 'info = os.lstat(cursor)' 'raw argv0 component lstat walk'
Contains-Token $bootstrap 'os.O_NOFOLLOW' 'bootstrap no-follow descriptors'
Contains-Token $bootstrap 'opened = os.fstat(descriptor)' 'verifier descriptor identity'
Contains-Token $bootstrap 'opened_package = os.fstat(package_descriptor)' 'package directory descriptor identity'
Contains-Token $bootstrap '_mount_identity_for_raw_absolute(raw)' 'entrypoint mount identity'
Contains-Token $bootstrap '_revalidate_raw_entrypoint_identity()' 'entrypoint identity revalidation'
Before-Token $bootstrap 'bootstrap = authenticate_package_before_imports()' 'np, common, kernels, overlay, tier_c_gate = _import_authenticated_modules()' 'bootstrap authentication precedes package imports'

Before-Token $trace 'if __name__ == "__main__":' 'import common' 'literal source-trace guard precedes package import'
Contains-Token $trace 'direct execution is forbidden' 'literal source-trace execution rejection'
Before-Token $runner 'if __name__ == "__main__":' 'import numpy as np' 'literal runner guard precedes numeric import'
Contains-Token $runner 'direct execution is forbidden' 'literal runner execution rejection'
Contains-Token $reuse 'standalone execution is forbidden' 'standalone reuse-helper rejection'

foreach($token in @(
    'def _assert_boundary_pair_disjoint(',
    'path boundary ancestry overlap',
    'path boundary device/inode alias',
    'path boundary mount-coordinate alias',
    'class BoundaryGuard:',
    'existing_component_identities',
    'nearest_ancestor_device',
    'nearest_ancestor_inode',
    'self._assert_all_disjoint(current_outputs, current_inputs)',
    'revalidation_required_before_every_create_new'
)){Contains-Token $common $token "boundary token $token"}
Contains-Token $trace 'guard = common.BoundaryGuard(' 'source-trace output boundary constructed'
Contains-Token $runner 'boundary = common.BoundaryGuard(' 'calibration output boundary constructed'
Contains-Token $runner 'boundary_guard = common.BoundaryGuard(' 'production output boundary constructed'
Before-Token $runner 'workspace_aux_closure = common.revalidate_workspace_aux_closure(' 'output_dir = common.ensure_output_directory(' 'workspace/aux closure precedes output root creation'
Before-Token $runner 'output_dir = common.ensure_output_directory(' 'journal = StateJournal(' 'output root creation precedes journal'
Contains-Token $runner 'production_boundary.revalidate("after output root create/open and before journal")' 'boundary revalidated before journal'

# Scientific replay, arithmetic and serialization guards inherited from v4.
foreach($token in @(
    'replayed = _compute_stage0_shard_state(',
    'replayed_pair = _compare_stage0_shard_replay(',
    'winner_ordinals, winner_q = _run_stage1_strict(',
    'completed_result, result_events_before_final = _prepare_completed_result_replay(',
    'return _commit_or_verify_result(result_path, journal, result, completed_result)',
    '_validate_full_stage0_q(access, q, len(candidate_ordinals))',
    'workspace_aux_closure = common.revalidate_workspace_aux_closure('
)){Contains-Token $runner $token "scientific replay token $token"}
foreach($productionSource in @($common,$runner,$trace)){
    Check (-not [regex]::IsMatch($productionSource,'(?m)^\s*assert\s')) 'production guards survive optimized mode'
}

$stage0=[int64]42205184*512
$stage1=[int64]135168*48624
$report=[int64]33*65536
Equal ([int64]$lock.search_cascade.stage0.maximum_generated_normal_values) $stage0 'stage0 arithmetic'
Equal ([int64]$lock.search_cascade.stage1.maximum_generated_normal_values) $stage1 'stage1 arithmetic'
Equal ([int64]$lock.search_cascade.post_selection_reporting_generated_normal_values) $report 'reporting arithmetic'
Equal ([int64]$lock.search_cascade.end_to_end_maximum_generated_normal_values) ($stage0+$stage1+$report) 'end-to-end arithmetic'
Equal ([int64]$lock.research_read_ledger.eligible_qwen_payload_bytes) ([int64]31*3145728) 'research-read arithmetic'
Equal ([int64]$lock.research_read_ledger.bound_v4_result_audit_event_topk_bytes) ([int64]2360016+9276+334+723930) 'bound metadata read arithmetic'
$side=[double]80*8/[double]28311552
Close ([double]$lock.physical_ledger.side_bpw) $side 1e-18 'side bpw arithmetic'
Close ([double]$lock.physical_ledger.maximum_compatible_base_codec_bpw_after_side_metadata) (2.15-$side) 1e-15 'base-codec cap arithmetic'
$appended=1.169444+20/([double]4718592*2.15/8)
Close ([double]$lock.physical_ledger.conservative_appended_cold_read_amplification_at_2_15_bpw) $appended 1e-15 'read amplification arithmetic'
Check ($appended -lt 2.0) 'read amplification below cap'

# Independent manifest mutation suite.
$manifestText=[Text.Encoding]::ASCII.GetString($manifestRaw)
$manifestLines=$manifestText.TrimEnd("`n").Split("`n")
$mutations=[ordered]@{}
$mutations.duplicate=[Text.Encoding]::ASCII.GetBytes(($manifestLines[0]+"`n"+$manifestText))
$swapped=@($manifestLines); $temp=$swapped[0];$swapped[0]=$swapped[1];$swapped[1]=$temp
$mutations.unsorted=[Text.Encoding]::ASCII.GetBytes(($swapped -join "`n")+"`n")
$mutations.uppercase=[Text.Encoding]::ASCII.GetBytes((($manifestLines[0].ToUpperInvariant())+"`n"+($manifestLines[1..($manifestLines.Count-1)]-join "`n")+"`n"))
$mutations.extra=[Text.Encoding]::ASCII.GetBytes($manifestText+('0'*64)+'  unexpected.py'+"`n")
$mutations.missing=[Text.Encoding]::ASCII.GetBytes(($manifestLines[1..($manifestLines.Count-1)]-join "`n")+"`n")
$mutations.crlf=[Text.Encoding]::ASCII.GetBytes($manifestText.Replace("`n","`r`n"))
$mutations.no_terminal_lf=[Text.Encoding]::ASCII.GetBytes($manifestText.TrimEnd("`n"))
foreach($entry in $mutations.GetEnumerator()){Check (Rejects-Manifest $entry.Value $expectedMembers) "manifest mutation rejects: $($entry.Key)"}

# Independent synthetic path-alias cases matching the frozen source grammar.
$baseOutput=@{path='/safe/out';exists=$false;device=$null;inode=$null;major_minor='8:1';filesystem_path='/safe/out'}
$ancestorInput=@{path='/safe';exists=$true;device=8;inode=10;major_minor='8:1';filesystem_path='/safe'}
$descendantInput=@{path='/safe/out/child';exists=$true;device=8;inode=11;major_minor='8:1';filesystem_path='/safe/out/child'}
$hardLeft=@{path='/a/out';exists=$true;device=8;inode=99;major_minor='8:1';filesystem_path='/a/out'}
$hardRight=@{path='/b/input';exists=$true;device=8;inode=99;major_minor='8:1';filesystem_path='/b/input'}
$bindLeft=@{path='/x/out';exists=$false;device=$null;inode=$null;major_minor='8:1';filesystem_path='/underlying/shared/out'}
$bindRight=@{path='/y/input';exists=$true;device=8;inode=101;major_minor='8:1';filesystem_path='/underlying/shared'}
$distinct=@{path='/other/input';exists=$true;device=9;inode=102;major_minor='9:2';filesystem_path='/other/input'}
Check (Synthetic-Pair-Rejects $baseOutput $ancestorInput) 'synthetic output-below-input ancestry rejects'
Check (Synthetic-Pair-Rejects $baseOutput $descendantInput) 'synthetic output-above-input ancestry rejects'
Check (Synthetic-Pair-Rejects $hardLeft $hardRight) 'synthetic device/inode alias rejects'
Check (Synthetic-Pair-Rejects $bindLeft $bindRight) 'synthetic mount-coordinate alias rejects'
Check (-not (Synthetic-Pair-Rejects $baseOutput $distinct)) 'synthetic disjoint paths pass'

# Fresh adversarial cases. These deliberately prove release blockers rather
# than asserting that the producer's own happy-path tests pass.
$bypassEvidence=[ordered]@{
    guard_condition='__name__ == __main__'
    imported_module_name='source_trace'
    guard_runs_when_imported=$false
    public_main_accepts_only_argv=$true
    verifier_passes_authentication_capability=$false
    receipt_binds_bootstrap_identity=$false
}
Check ($trace.Contains('def main(argv: Sequence[str] | None = None) -> int:',[StringComparison]::Ordinal)) 'source trace exposes public argv-only main'
Check ($bootstrap.Contains('return source_trace.main(dispatch_args)',[StringComparison]::Ordinal)) 'verifier calls public source-trace main without capability'
Check (-not $trace.Contains('dispatch_capability',[StringComparison]::Ordinal)) 'source trace has no authenticated dispatch capability'
Check (-not $testTrace.Contains('import_and_call',[StringComparison]::Ordinal)) 'producer suite lacks import-call bypass regression'

$atomicTokens=@('dir_fd=','openat2','O_PATH','renameat','fstatat','fs-verity')
foreach($token in $atomicTokens){Check (-not $common.Contains($token,[StringComparison]::Ordinal)) "atomic boundary primitive absent: $token"}
Before-Token $trace 'guard.revalidate("immediately before source-trace create-new")' 'output = common.write_json_create_new(' 'source-trace revalidate and create are separate calls'
Check ([regex]::IsMatch(
    $runner,
    'boundary\.revalidate\("immediately before calibration create-new"\)\s+common\.write_json_create_new\(',
    [Text.RegularExpressions.RegexOptions]::CultureInvariant
)) 'calibration revalidate and create are separate calls'
Contains-Token $common 'descriptor = os.open(absolute, flags, 0o600)' 'create-new reopens output by pathname'
Contains-Token $common 'def open_create_new(path: Path, *, binary: bool, label: str):' 'common create helper receives no boundary guard'
$mountSwapEvidence=[ordered]@{
    t0='boundary revalidation sees a distinct output mount coordinate'
    t1='the output ancestor is atomically replaced by a bind alias after revalidation'
    t2='path-based create-new follows the replacement because only final O_NOFOLLOW is used'
    guard_and_create_atomic=$false
    post_create_boundary_check_in_source_trace=$false
}

Before-Token $bootstrap 'bootstrap = authenticate_package_before_imports()' 'np, common, kernels, overlay, tier_c_gate = _import_authenticated_modules()' 'authenticated closure and import are separate operations'
Contains-Token $bootstrap 'os.close(descriptor)' 'authenticated file descriptors close before imports'
Check (-not $bootstrap.Contains('SourceFileLoader',[StringComparison]::Ordinal)) 'no authenticated-byte import loader'
Check (-not $bootstrap.Contains('sys.meta_path',[StringComparison]::Ordinal)) 'no authenticated-byte meta-path loader'
$transientImportEvidence=[ordered]@{
    t0='manifest hashes authenticated bytes and closes descriptors'
    t1='same-inode same-size module content changes before pathname import'
    t2='pathname import executes transient bytes'
    t3='content is restored before post-import hash replay'
    identity_and_both_hash_checks_can_pass=$true
    authenticated_bytes_are_the_executed_bytes=$false
}

$adversarial=[ordered]@{
    manifest_mutations_rejected=@($mutations.Keys)
    lexical_ancestry_rejected=$true
    hardlink_inode_alias_rejected=$true
    bind_mount_coordinate_alias_rejected_at_check_time=$true
    unauthenticated_import_call_bypass=$bypassEvidence
    mount_swap_between_revalidation_and_create=$mountSwapEvidence
    transient_authenticated_package_import_swap=$transientImportEvidence
}
$adversarialJson=$adversarial|ConvertTo-Json -Depth 20 -Compress
$evidenceSha=Sha-Text $adversarialJson

$evidenceLines=[ordered]@{
    bootstrap_raw_argv_capture=(Line-Of $bootstrap 'raw = sys.argv[0]')
    bootstrap_authenticated_import_gap=(Line-Of $bootstrap 'np, common, kernels, overlay, tier_c_gate = _import_authenticated_modules()')
    source_trace_direct_guard=(Line-Of $trace 'if __name__ == "__main__":')
    source_trace_public_main=(Line-Of $trace 'def main(argv: Sequence[str] | None = None) -> int:')
    source_trace_separate_revalidate=(Line-Of $trace 'guard.revalidate("immediately before source-trace create-new")')
    source_trace_path_create=(Line-Of $trace 'output = common.write_json_create_new(')
    common_boundary_guard=(Line-Of $common 'class BoundaryGuard:')
    common_path_open=(Line-Of $common 'descriptor = os.open(absolute, flags, 0o600)')
    runner_workspace_closure=(Line-Of $runner 'workspace_aux_closure = common.revalidate_workspace_aux_closure(')
    runner_output_create=(Line-Of $runner 'output_dir = common.ensure_output_directory(')
    runner_journal_create=(Line-Of $runner 'journal = StateJournal(')
}

$result=[ordered]@{
    schema='tier_c_grouped_v5_layout_overlay_v5_independent_source_audit_execution_v1'
    status='BLOCKED_AUTHENTICATED_DISPATCH_AND_ATOMIC_TOCTOU_REPAIRS_REQUIRED'
    producer_manifest_sha256='e9ef19853d1350ac4085a18a65b19a6805620ef7a091447ce57404715f88805f'
    producer_manifest_rows=16
    exact_regular_non_link_closure='17_OF_17'
    candidate_lock_file_sha256='cec0b12927340d82c1c8c78cd02b7849c7371b1b16d7f2c1b1d24ae889ada58d'
    candidate_lock_internal_sha256=$internalSeal
    dependency_hashes='16_OF_16'
    assertions_passed=$script:Checks
    v4_repair_findings=[ordered]@{
        raw_argv0_component_inode_mount_identity='PASS_AT_CHECK_INSTANTS'
        literal_direct_script_execution='PASS'
        authenticated_dispatch_exclusivity='BLOCK_IMPORT_CALL_BYPASS'
        lexical_inode_mount_disjointness='PASS_AT_CHECK_INSTANTS'
        workspace_aux_closure_before_output_and_journal='PASS'
        atomic_toctou_closure='BLOCK_PATH_REOPEN_GAPS'
        authenticated_bytes_equal_executed_bytes='BLOCK_TRANSIENT_IMPORT_GAP'
    }
    arithmetic=[ordered]@{
        stage0_generated_values=$stage0
        stage1_generated_values=$stage1
        reporting_generated_values=$report
        end_to_end_generated_values=$stage0+$stage1+$report
        side_bpw=$side
        appended_read_amplification=$appended
    }
    adversarial_evidence=$adversarial
    adversarial_evidence_sha256=$evidenceSha
    source_line_evidence=$evidenceLines
    access=[ordered]@{
        payload_paths_supplied=0
        payload_files_opened=0
        payload_manifest_or_directory_operations=0
        python_imports=0
        forbidden_runtime_imports=0
        accelerator_jobs=0
        network_operations=0
        production_runs=0
        producer_files_modified=0
    }
    verdict='BLOCK_DO_NOT_AUTHORIZE_CALIBRATION_OR_PRODUCTION'
}
$result|ConvertTo-Json -Depth 30
