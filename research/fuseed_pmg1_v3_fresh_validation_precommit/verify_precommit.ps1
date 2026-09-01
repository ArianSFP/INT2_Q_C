$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Lock = Get-Content -LiteralPath (Join-Path $Root 'validation_precommit.json') -Raw | ConvertFrom-Json -AsHashtable

if ($Lock.schema -ne 'fuseed_pmg1_v3_fresh_validation_precommit_v1' -or -not $Lock.sealed) { throw 'schema/seal mismatch' }
if ($Lock.firewall.payload_access_authorized -or $Lock.firewall.execution_authorized) { throw 'precommit grants authority' }

$ranked = foreach ($e in 0..127) {
    if (($e % 8) -eq 0) { continue }
    $wire = [Text.Encoding]::UTF8.GetBytes("FUSEED-PMG1-v3|fresh-validation|layer=15|expert=$e")
    [pscustomobject]@{ expert = $e; hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($wire)).ToLowerInvariant() }
}
$top = @($ranked | Sort-Object hash | Select-Object -First 4)
$expectedExperts = @(67, 95, 69, 34)
$expectedHashes = @(
    '034d9a6fd2d7f1c53c5a17e71f1a7b4dcbd61780388a63491822f9780204aa9a',
    '03fb4a2abe45c953a93ef17c92ca49e4a785c232fbe0ac64d9777a5cdc34dcaf',
    '06ca5c34443be9a2dbcf6bb2a597e37844204ef884f82439d89e7262e2f0bb5f',
    '0a146d6a79c0d5daa7ac233234d8add354cc710c639e11a598bbb2cf36d934dd'
)
for ($i = 0; $i -lt 4; $i++) {
    if ($top[$i].expert -ne $expectedExperts[$i] -or $top[$i].hash -ne $expectedHashes[$i]) { throw "rank mismatch at $i" }
    if ($Lock.selection_rule.selected_in_rank_order[$i] -ne $expectedExperts[$i]) { throw "locked expert mismatch at $i" }
    if ($Lock.selection_rule.selected_hashes_in_rank_order[$i] -ne $expectedHashes[$i]) { throw "locked hash mismatch at $i" }
}

$metadataRoot = (Resolve-Path -LiteralPath (Join-Path $Root '..\..\..\qwen_weight_cache')).Path
$indexPath = Join-Path $metadataRoot 'model.safetensors.index.json'
$indexHash = (Get-FileHash -LiteralPath $indexPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($indexHash -ne $Lock.public_metadata_bindings.index_sha256) { throw 'index hash mismatch' }
$index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json

$headers = @{}
foreach ($binding in $Lock.public_metadata_bindings.headers) {
    $name = Split-Path -Leaf $binding.path
    $path = Join-Path (Join-Path $metadataRoot 'headers') $name
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $binding.sha256) { throw "header hash mismatch: $name" }
    $wrapper = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if ([int64]$wrapper.header_length -ne [int64]$binding.header_length -or (8 + [int64]$wrapper.header_length) -ne [int64]$binding.data_start) { throw "header length mismatch: $name" }
    $shard = $name -replace '\.header\.json$', ''
    $headers[$shard] = $wrapper
}

if ($Lock.tensors.Count -ne 12) { throw 'tensor cardinality mismatch' }
$seen = @{}
foreach ($tensor in $Lock.tensors) {
    if ($seen.ContainsKey($tensor.name)) { throw "duplicate tensor: $($tensor.name)" }
    $seen[$tensor.name] = $true
    if (($tensor.expert -notin $expectedExperts) -or ($tensor.role -notin @('gate','up','down'))) { throw 'tensor identity outside precommit' }
    $mapped = ($index.weight_map.PSObject.Properties | Where-Object Name -EQ $tensor.name).Value
    if ($mapped -ne $tensor.shard) { throw "index mapping mismatch: $($tensor.name)" }
    $meta = ($headers[$tensor.shard].header.PSObject.Properties | Where-Object Name -EQ $tensor.name).Value
    if ($null -eq $meta -or $meta.dtype -ne 'BF16') { throw "metadata missing/dtype mismatch: $($tensor.name)" }
    if (($meta.shape -join ',') -ne ($tensor.shape -join ',')) { throw "shape mismatch: $($tensor.name)" }
    if (($meta.data_offsets -join ',') -ne ($tensor.relative_offsets -join ',')) { throw "offset mismatch: $($tensor.name)" }
    $binding = @($Lock.public_metadata_bindings.headers | Where-Object { (Split-Path -Leaf $_.path) -eq ($tensor.shard + '.header.json') })[0]
    $start = [int64]$binding.data_start + [int64]$tensor.relative_offsets[0]
    $stop = [int64]$binding.data_start + [int64]$tensor.relative_offsets[1] - 1
    if ($start -ne [int64]$tensor.http_range_inclusive[0] -or $stop -ne [int64]$tensor.http_range_inclusive[1]) { throw "HTTP range mismatch: $($tensor.name)" }
    if (($stop - $start + 1) -ne 3145728) { throw "tensor byte count mismatch: $($tensor.name)" }
}

Write-Output 'PASS: PMG1-v3 fresh validation experts 67,95,69,34 and twelve metadata-only tensor ranges are deterministically precommitted; payload and execution remain blocked.'
