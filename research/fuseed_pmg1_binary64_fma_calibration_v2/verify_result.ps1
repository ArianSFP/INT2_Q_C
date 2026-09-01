$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$research = Split-Path -Parent $root
$resultPath = Join-Path $root 'result.json'
$result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json -Depth 100

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "VERIFY FAILED: $Message" }
}
function Close-Enough([double]$Actual, [double]$Expected) {
    $scale = [Math]::Max(1.0, [Math]::Abs($Expected))
    return [Math]::Abs($Actual - $Expected) -le 1e-12 * $scale
}
function Sha([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}
function Bytes-Sha([byte[]]$Bytes) {
    [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}

$scriptPath = Join-Path $root 'calibrate_fma.py'
$templatePath = Join-Path $research 'fuseed_pmg1_binary64_calibration_v1\calibrate_binary64.py'
$designManifest = Join-Path $research 'fuseed_pmg1_v2_design_draft\ARTIFACT_SHA256SUMS.txt'
Require ((Sha $scriptPath) -eq '0e2f354415d2d8cfebfceda58b6ade77eddc2b4e025488baba329beba09d0a87') 'wrapper hash mismatch'
Require ((Sha $templatePath) -eq '9376720ec812b93e070ccb93433e83ff243213d6d244c7a18afa84b3d8690c24') 'template hash mismatch'
Require ((Sha $designManifest) -eq 'ea7086c401cd6981d097ecc9b52196d3d01cda123d0cb8ab28c001cf008b27ff') 'design manifest mismatch'
Require ((Sha $resultPath) -eq '82e29cbfc8ec1ac23761c37712a3fda3d2745b04c9a71ae296ce864796ddc75e') 'result file hash mismatch'

Require ($result.schema -eq 'fuseed_pmg1_binary64_explicit_fma_stage0_calibration_v2') 'schema mismatch'
Require ($result.status -eq 'EXPLICIT_FMA_BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT') 'status mismatch'
Require ($result.script_sha256 -eq (Sha $scriptPath)) 'result script binding mismatch'
Require ($result.bindings.design_manifest.sha256 -eq (Sha $designManifest)) 'result design-manifest binding mismatch'
Require ($result.bindings.design_complete_plan_sha256 -eq '86639758eda1835b9ea9e883372bb55ec13ec3487705a91d892878972db74760') 'complete plan digest mismatch'
Require ($result.bindings.design_bundle_sha256 -eq '16aacb6f5fa6a1ed12fe0c01506410ad69585894077a4a6af627674b6b90adda') 'bundle digest mismatch'
Require ($result.derivation.abi_id -eq 'CURRENT_PMG_GATE_UP_DIRECT_BF16' -and $result.derivation.active_domains -eq 1) 'ABI/domain mismatch'
Require ($result.derivation.metric_wire -eq 'IEEE-754 binary64 capture, canonical positive zero') 'metric wire mismatch'
Require ($result.derivation.metric_order -eq 'capture descending then seed_u32 ascending') 'metric order mismatch'
$counts = $result.derivation.binary64_capture_replacement_counts
foreach ($name in @('metric_function_signature','metric_return','output_pointer','centered_moments_rn','mu_rn','sse_explicit_fma','baseline_explicit_fma','anchor_square_explicit_fma','domain0_cross_explicit_fma','domain1_cross_explicit_fma','capture_rn')) {
    Require ([int]$counts.$name -eq 1) "replacement count mismatch: $name"
}
Require ([int]$counts.metric_call_sites_plus_definition -eq 2) 'metric call count mismatch'
Require ([int]$counts.explicit_fma_source_occurrences -eq 10) 'explicit FMA occurrence mismatch'
Require ([int]$counts.rounding_intrinsic_source_occurrences -eq 19) 'rounding intrinsic occurrence mismatch'

$options = @('--std=c++17','--fmad=true','--ftz=false','--prec-div=true','--prec-sqrt=true','-I/usr/local/cuda/include')
foreach ($kind in @('performance','parity')) {
    $compiled = $result.compiled_kernels.$kind
    Require (($compiled.options -join '|') -eq ($options -join '|')) "compile options mismatch: $kind"
    Require ($compiled.arch -eq '120' -and $compiled.cubin_magic_hex -eq '7f454c4602010141') "compiled identity mismatch: $kind"
}
Require ($result.compiled_kernels.performance.cubin_sha256 -eq '41e71c07819ac6ce99e0bfb4c3903aa8400e20fa955ce3157e215a7d732b55ac') 'performance cubin mismatch'
Require ($result.compiled_kernels.parity.cubin_sha256 -eq 'd56bac00339f3fd6a25e903f15f85abc6972fcd094c3ec64238acb8378e781f8') 'parity cubin mismatch'
Require ($result.compiled_kernels.parity_source_sha256 -eq '63cc615c7f9b0a5f07920cc6c9b04516160e3784d3dabf5427f170efe3bd46b1') 'parity source mismatch'

Require ($result.runtime.python -eq '3.12.3' -and $result.runtime.numpy -eq '2.5.2' -and $result.runtime.cupy -eq '14.2.0') 'Python runtime version mismatch'
Require ($result.runtime.torch -eq '2.8.0+cu128' -and $result.runtime.torch_cuda -eq '12.8') 'Torch version mismatch'
Require ($result.runtime.cuda_runtime -eq 12090 -and $result.runtime.cuda_driver_api -eq 13000) 'CUDA version mismatch'
Require (($result.runtime.nvrtc -join '.') -eq '12.8') 'NVRTC version mismatch'
Require ($result.runtime.device -eq 'NVIDIA GeForce RTX 5090' -and $result.runtime.compute_capability -eq '120') 'GPU identity mismatch'
Require ($result.runtime.cuda_visible_devices -eq '0' -and $result.runtime.pythonpath -eq '/usr/local/lib/python3.12/dist-packages') 'runtime environment mismatch'
Require (@($result.runtime.loaded_cuda_libraries).Count -eq 3) 'loaded CUDA library closure mismatch'
$expectedRuntimeHashes = @{
    '/usr/bin/python3.12'='1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5'
    '/workspace/int2-cupy-venv/lib/python3.12/site-packages/numpy/__init__.py'='09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11'
    '/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy/__init__.py'='8c4724758587dea5f1c1d7c217c74a9fa0e4ed7f9d76a2b86fa001117cf3c718'
    '/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy/cuda/compiler.py'='09226d26ab41bf6e7b5b6e57b59187b4c3a5637690747af9a83d288a87d0fb6e'
    '/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy_backends/cuda/libs/nvrtc.cpython-312-x86_64-linux-gnu.so'='a3e9213226fa693231cab5e873aa1de8d31f7c6d82d9c56716c326ce438af373'
    '/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy_backends/cuda/api/runtime.cpython-312-x86_64-linux-gnu.so'='65a5c75db5e05c9bd35132b7f41631cadff6a6a6300acd85b273db3ba7ce28de'
    '/usr/local/lib/python3.12/dist-packages/torch/__init__.py'='2f0deb66d5dff6b9c02a62832c3bf3824c2ee031c462a3afeb9ca170466da5bf'
    '/usr/local/lib/python3.12/dist-packages/torch/_C.cpython-312-x86_64-linux-gnu.so'='db1e4f96208c6b297186585a04acee533035705706254ebdd4953fbae6b90224'
    '/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'='43731e24cd89e3749826304f304e8aa11fbecf1188715271b1f5018d6212b5e6'
    '/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12'='c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920'
    '/usr/lib/x86_64-linux-gnu/libcuda.so.580.126.09'='e8e541166449da5a1278f40b27a28d072174b31f2941b101a9609b6d1d3aed32'
}
Require (@($result.runtime.file_bindings.PSObject.Properties).Count -eq $expectedRuntimeHashes.Count) 'runtime file-binding count mismatch'
foreach ($property in $result.runtime.file_bindings.PSObject.Properties) {
    Require ($expectedRuntimeHashes.ContainsKey($property.Name)) "unexpected runtime file: $($property.Name)"
    Require ($property.Value.sha256 -eq $expectedRuntimeHashes[$property.Name]) "runtime file receipt mismatch: $($property.Name)"
}

$p0 = $result.parity.direct_shifted_reference_three_replays
$p1 = $result.parity.direct_shifted_and_original_offset_sequential_three_replays
$pt = $result.parity.torch_initial_terminal_and_bf16_three_replays
Require ($p0.repetitions -eq 3 -and $p0.identical -eq $true) 'direct parity replay mismatch'
Require ($p0.receipt.rows -eq 132 -and $p0.receipt.raw_bitwise_equal -eq $true -and $p0.receipt.scaled_bf16_bitwise_equal -eq $true -and $p0.receipt.terminal_counter_equal -eq $true) 'direct parity receipt mismatch'
Require ($p1.repetitions -eq 3 -and $p1.identical -eq $true -and $p1.receipt.rows -eq 132) 'sequential parity replay mismatch'
Require ($p1.receipt.direct_equals_shifted -eq $true -and $p1.receipt.direct_equals_original_offset_sequential_j_plus_1 -eq $true -and $p1.receipt.terminal_counters_equal -eq $true) 'dual-reference parity mismatch'
Require ($pt.repetitions -eq 3 -and $pt.identical -eq $true -and $pt.case_count -eq 8 -and $pt.coordinate_count -eq 56 -and $pt.stride -eq 261120) 'Torch parity panel mismatch'
Require (@($pt.rows).Count -eq 8) 'Torch parity row count mismatch'
foreach ($row in $pt.rows) {
    Require ($row.initial_seed -eq $row.effective_seed) 'Torch effective seed mismatch'
    Require ($row.initial_offset -eq $row.offset) 'Torch initial offset mismatch'
    Require ($row.terminal_offset -eq ($row.initial_offset + $row.expected_increment)) 'Torch terminal offset mismatch'
    Require ($row.initial_state_sha256 -match '^[0-9a-f]{64}$' -and $row.terminal_state_sha256 -match '^[0-9a-f]{64}$') 'Torch state hash malformed'
}

Require ($result.shape.candidates_per_shard -eq 16777216 -and $result.shape.shards -eq 256 -and $result.shape.complete_candidate_count -eq 4294967296) 'candidate partition mismatch'
Require ($result.shape.abi_count -eq 1 -and $result.shape.active_domains -eq 1) 'shape ABI/domain mismatch'
Require ($result.shape.q_dtype -eq 'binary64' -and [uint64]$result.shape.q_bytes -eq 134217728) 'binary64 q ledger mismatch'
Require ($result.shape.journal_record_bytes -eq 12 -and [uint64]$result.shape.all_shard_topk_record_bytes -eq 25165824) 'packed journal ledger mismatch'
Require ($result.shape.top_k -eq 8192 -and $result.shape.repetitions -eq 3) 'Top-K/repetition mismatch'
Require (@($result.rows).Count -eq 3) 'timing row count mismatch'
Require (@($result.rows.topk_seed_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K seed replays differ'
Require (@($result.rows.topk_capture_sha256 | Select-Object -Unique).Count -eq 1) 'Top-K capture replays differ'
Require (@($result.rows.q_sentinel_sha256 | Select-Object -Unique).Count -eq 1) 'q replays differ'

$expectedJournalHashes = @(
    'b5d640d451bd39c1d18e2d319d1e1c74a0060c5b255df5edb73adf7da555f997',
    '19ea5c141ab0be9fe74d3585b47c3f25855b4335fc07bf9ca6af2b7a901b1ee5',
    '2737b32808dd2b3698c670ea73be235ff1742950507fab6e63320b51996ad32b'
)
for ($repetition = 0; $repetition -lt 3; $repetition++) {
    $row = $result.rows[$repetition]
    Require ($row.repetition -eq $repetition -and $row.negative_zero_count -eq 0) "row identity/zero mismatch: $repetition"
    $journalPath = Join-Path $root "binary64_shard_replay_$repetition.bin"
    Require ((Sha $journalPath) -eq $expectedJournalHashes[$repetition]) "journal file hash mismatch: $repetition"
    Require ($row.journal_sha256 -eq $expectedJournalHashes[$repetition]) "journal receipt hash mismatch: $repetition"
    $bytes = [System.IO.File]::ReadAllBytes($journalPath)
    $headerLength = [BitConverter]::ToUInt32($bytes,0)
    $recordOffset = 4 + [int]$headerLength
    Require (($bytes.Length - $recordOffset) -eq 8192*12) "journal record span mismatch: $repetition"
    $headerText = [Text.Encoding]::UTF8.GetString($bytes,4,[int]$headerLength)
    $header = $headerText | ConvertFrom-Json -Depth 20
    Require ($header.schema -eq 'fuseed_pmg1_binary64_explicit_fma_shard_journal_v2') "journal schema mismatch: $repetition"
    Require ($header.repetition -eq $repetition -and $header.top_k -eq 8192 -and $header.candidate_count -eq 16777216) "journal header shape mismatch: $repetition"
    Require ($header.record_wire -eq 'packed little-endian u32 seed then binary64 capture') "journal wire mismatch: $repetition"
    Require ($header.metric_order -eq 'capture descending then seed_u32 ascending') "journal order mismatch: $repetition"
    $seedBytes = [byte[]]::new(8192*4)
    $captureBytes = [byte[]]::new(8192*8)
    [uint32]$priorSeed = 0
    [double]$priorCapture = [double]::PositiveInfinity
    for ($index=0; $index -lt 8192; $index++) {
        $offset = $recordOffset + 12*$index
        [Buffer]::BlockCopy($bytes,$offset,$seedBytes,4*$index,4)
        [Buffer]::BlockCopy($bytes,$offset+4,$captureBytes,8*$index,8)
        [uint32]$seed = [BitConverter]::ToUInt32($bytes,$offset)
        [double]$capture = [BitConverter]::ToDouble($bytes,$offset+4)
        Require (-not [double]::IsNaN($capture) -and -not [double]::IsInfinity($capture)) "nonfinite journal capture: $repetition/$index"
        if ($index -gt 0) {
            Require ($capture -lt $priorCapture -or ($capture -eq $priorCapture -and $seed -ge $priorSeed)) "journal total order mismatch: $repetition/$index"
        }
        $priorSeed=$seed; $priorCapture=$capture
    }
    Require ((Bytes-Sha $seedBytes) -eq $row.topk_seed_sha256) "journal seed payload mismatch: $repetition"
    Require ((Bytes-Sha $captureBytes) -eq $row.topk_capture_sha256) "journal capture payload mismatch: $repetition"
    Require ($header.seed_sha256 -eq $row.topk_seed_sha256 -and $header.capture_sha256 -eq $row.topk_capture_sha256) "journal header payload hash mismatch: $repetition"
}

$times = @($result.rows.shard_end_to_end_seconds | Sort-Object)
$median = [double]$times[1]
$cold = [Math]::Max(0.0, [double]$result.rows[0].shard_end_to_end_seconds - $median)
$projection = $median*256.0 + $cold + [double]$result.global_merge_shape_probe.seconds
Require (Close-Enough $median ([double]$result.aggregate.median_complete_stage0_shard_seconds)) 'median shard mismatch'
Require (Close-Enough $cold ([double]$result.aggregate.one_time_cold_excess_seconds)) 'cold excess mismatch'
Require (Close-Enough $projection ([double]$result.aggregate.projected_complete_u32_stage0_seconds_including_finite_topk_journal_and_global_merge)) 'projection arithmetic mismatch'
Require (Close-Enough $projection 520.8358833260136) 'frozen projection mismatch'
Require ($result.aggregate.prospective_stage0_margin_gate_seconds -eq 650.0 -and $projection -lt 650.0 -and $result.aggregate.stage0_projection_below_margin_gate -eq $true) '650-second gate mismatch'
Require ($result.aggregate.full_pipeline_wall_gate_seconds -eq 900.0 -and $result.aggregate.full_pipeline_projection_claimed -eq $false) 'full-pipeline claim boundary mismatch'
Require ($result.access.model_or_qwen_path_arguments -eq 0 -and $result.access.payload_files_opened -eq 0 -and $result.access.network_operations -eq 0) 'access attestation mismatch'

Write-Output 'PASS: explicit-FMA binary64 PMG1 stage0 result, cubins, parity, packed journals, and 520.8358833260136-second projection are internally consistent; full pipeline and payload remain unauthorized.'
