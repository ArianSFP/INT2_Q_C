[CmdletBinding()]
param(
    [string]$Package = $PSScriptRoot,
    [switch]$PrintPlanFacts,
    [switch]$PrintPlanJson
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw 'VERIFY FAILED: pwsh 7 or newer is required'
}

$script:Checks = [int64]0
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "VERIFY FAILED: $Message" }
    $script:Checks++
}
function Assert-Eq($Actual, $Expected, [string]$Message) {
    if ($Actual -ne $Expected) {
        throw "VERIFY FAILED: $Message; actual=$Actual expected=$Expected"
    }
    $script:Checks++
}
function Assert-Close([double]$Actual, [double]$Expected, [double]$Tolerance, [string]$Message) {
    if ([math]::Abs($Actual - $Expected) -gt $Tolerance) {
        throw "VERIFY FAILED: $Message; actual=$Actual expected=$Expected tolerance=$Tolerance"
    }
    $script:Checks++
}
function File-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Bytes-Sha256([byte[]]$Bytes) {
    [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}
function Text-Sha256([string]$Text) {
    Bytes-Sha256 ([System.Text.Encoding]::UTF8.GetBytes($Text))
}
function Hash-Text([string]$Text) {
    [System.Security.Cryptography.SHA256]::HashData([System.Text.Encoding]::UTF8.GetBytes($Text))
}
function U32-LE([byte[]]$Bytes, [int]$Offset) {
    [BitConverter]::ToUInt32($Bytes, $Offset)
}
function Canonical-Key([int]$Expert, [string]$Role, [int]$Row, [int]$Column) {
    'e{0:D3}|{1}|r{2:D3}|c{3:D4}' -f $Expert,$Role,$Row,$Column
}
function HashSet-Ordinal {
    $set = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    return ,$set
}
function Digest-Lines([System.Collections.IEnumerable]$Lines) {
    $array = @($Lines)
    Text-Sha256 (($array -join "`n") + "`n")
}

Assert-True ([BitConverter]::IsLittleEndian) 'verifier host is little-endian'
$root = [System.IO.Path]::GetFullPath($Package)
$repo = [System.IO.Path]::GetFullPath((Join-Path $root '..\..'))
$designPath = Join-Path $root 'design_lock.json'
$bindingsPath = Join-Path $root 'source_bindings.json'
$receiptPath = Join-Path $root 'DESIGN_RECEIPT.json'
Assert-True (Test-Path -LiteralPath $designPath -PathType Leaf) 'design lock exists'
Assert-True (Test-Path -LiteralPath $bindingsPath -PathType Leaf) 'source bindings exist'
Assert-True (Test-Path -LiteralPath $receiptPath -PathType Leaf) 'design receipt exists'
$design = Get-Content -LiteralPath $designPath -Raw | ConvertFrom-Json -Depth 100
$bindings = Get-Content -LiteralPath $bindingsPath -Raw | ConvertFrom-Json -Depth 100
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json -Depth 100

Assert-Eq ([string]$design.schema) 'fuseed_pmg1_v2_source_only_design_draft_v1' 'schema'
Assert-Eq ([string]$design.status) 'SOURCE_ONLY_DESIGN_DRAFT_CONDITIONALLY_DEFENSIBLE_EXECUTION_BLOCKED' 'status'
Assert-Eq ([bool]$design.sealed) $false 'draft is not a frozen launch protocol'
Assert-Eq ([bool]$design.authorization.implementation_or_execution_authorized) $false 'no execution authorization'
Assert-Eq ([string]$bindings.schema) 'fuseed_pmg1_v2_source_bindings_draft_v1' 'bindings schema'
Assert-Eq ([string]$bindings.status) 'SOURCE_ONLY_DEPENDENCY_CLOSURE_NO_EXECUTION_AUTHORITY' 'bindings status'
Assert-Eq ([string]$receipt.schema) 'fuseed_pmg1_v2_source_only_design_receipt_v1' 'receipt schema'
Assert-Eq ([string]$receipt.status) 'PASS_DESIGN_LOGIC_CONDITIONALLY_DEFENSIBLE_EXECUTION_BLOCKED' 'receipt status'
Assert-Eq ([string]$receipt.authorization) 'NONE' 'receipt authorization'
Assert-Eq @($bindings.dependencies).Count 11 'bound dependency count'
foreach ($dependency in $bindings.dependencies) {
    $dependencyPath = [System.IO.Path]::GetFullPath((Join-Path $root ([string]$dependency.path)))
    Assert-True ($dependencyPath.StartsWith($repo,[StringComparison]::OrdinalIgnoreCase)) "dependency remains in repository: $($dependency.id)"
    Assert-True (Test-Path -LiteralPath $dependencyPath -PathType Leaf) "dependency exists: $($dependency.id)"
    Assert-Eq (File-Sha256 $dependencyPath) ([string]$dependency.sha256) "dependency hash: $($dependency.id)"
}

$v1DesignPath = Join-Path $repo 'research\fuseed_u32_v1\design_lock.json'
$v1VerifierPath = Join-Path $repo 'research\fuseed_u32_v1\verify_design.ps1'
Assert-Eq (File-Sha256 $v1DesignPath) '7524a1dfc906dc8bb9addec5a624d16fdee068b01b8ed54ef118bcbf07d7eb48' 'bound v1 design bytes'
Assert-Eq (File-Sha256 $v1VerifierPath) '48fd85b716aafa9097c65919d51becb449b24e5c7497fc0e8f6806bd32a99511' 'bound v1 verifier bytes'
$v1 = Get-Content -LiteralPath $v1DesignPath -Raw | ConvertFrom-Json -Depth 100

$family = $v1.frozen_family
Assert-Eq ([string]$family.abis[1].abi_id) 'CURRENT_PMG_GATE_UP_DIRECT_BF16' 'only chosen ABI'
Assert-Eq ([int64]$family.abis[1].up_numel) ([int64]3145728) 'chosen ABI up call size'
Assert-Eq ([int64]$family.abis[1].down_numel) ([int64]1572864) 'chosen ABI down call size'
Assert-Eq ([string]$design.candidate_family.abi_id) ([string]$family.abis[1].abi_id) 'draft ABI matches bound v1 ABI1'
Assert-Eq ([int64]$design.candidate_family.candidate_count) ([int64]4294967296) 'one complete u32 family'

# Prove complete, disjoint 2^24 sharding of the u32 label space.
$seedCount = [int64]4294967296
$shardSize = [int64]16777216
$shardCount = 256
$lastEnd = [int64]0
for ($shard = 0; $shard -lt $shardCount; $shard++) {
    $start = [int64]$shard * $shardSize
    $end = $start + $shardSize
    Assert-Eq $start $lastEnd "shard $shard is contiguous"
    Assert-Eq ($end - $start) $shardSize "shard $shard size"
    Assert-True ($start -ge 0 -and $end -le $seedCount) "shard $shard bounds"
    $lastEnd = $end
}
Assert-Eq $lastEnd $seedCount 'shards exhaust u32 exactly once'

# The following is an independent stdlib re-enumeration of the exact frozen v1
# plan. ABI0 and ABI2 are enumerated only because ABI1's accepted coordinates
# were frozen under v1's cross-ABI opposite-split collision rule. Only ABI1 is
# emitted or admitted as a v2 candidate family.
$stride = [int64]261120
$selectionExperts = @(0,8,16,32,40,48,64,72,80,96,104,112)
$validationExperts = @(24,56,88,120)
$identities = [System.Collections.Generic.List[object]]::new()
foreach ($expert in $selectionExperts) {
    foreach ($role in @('up','down')) {
        if ($expert -eq 0 -and $role -eq 'up') { continue }
        $identities.Add([pscustomobject]@{Expert=[int]$expert;Role=[string]$role})
    }
}
$validationIdentities = [System.Collections.Generic.List[object]]::new()
foreach ($expert in $validationExperts) {
    foreach ($role in @('up','down')) {
        $validationIdentities.Add([pscustomobject]@{Expert=[int]$expert;Role=[string]$role})
    }
}
Assert-Eq $identities.Count 23 'selection identity count'
Assert-Eq $validationIdentities.Count 8 'validation identity count'

$highCount = HashSet-Ordinal
foreach ($spec in @(@{Role='up';Count=6},@{Role='down';Count=7})) {
    $ranked = @($identities | Where-Object Role -eq $spec.Role | ForEach-Object {
        $text = "FUSEED-U32-v1|quota|$($_.Expert)|$($_.Role)"
        [pscustomobject]@{Key=(Canonical-Key $_.Expert $_.Role 0 0);Id=$_;Hash=(Text-Sha256 $text)}
    } | Sort-Object Hash,Key)
    for ($index=0; $index -lt $spec.Count; $index++) {
        [void]$highCount.Add("$($ranked[$index].Id.Expert)|$($ranked[$index].Id.Role)")
    }
}
Assert-Eq $highCount.Count 13 'high-count identity count'

function Invert-Native([int]$Abi, [int]$WantedExpert, [string]$WantedRole, [int64]$Native) {
    if ($Abi -eq 0) {
        if ($WantedRole -eq 'up') {
            [int64]$block = 2048*1536
            $localExpert = [int][math]::Floor($Native/$block)
            $within = $Native % $block
            $column = [int][math]::Floor($within/1536)
            $fusedRow = [int]($within % 1536)
            if ($localExpert -ne ($WantedExpert % 32) -or $fusedRow -lt 768) { return $null }
            return [pscustomobject]@{Row=$fusedRow-768;Column=$column}
        }
        [int64]$block = 768*2048
        $localExpert = [int][math]::Floor($Native/$block)
        $within = $Native % $block
        $row = [int][math]::Floor($within/2048)
        $column = [int]($within % 2048)
        if ($localExpert -ne ($WantedExpert % 32)) { return $null }
        return [pscustomobject]@{Row=$row;Column=$column}
    }
    if ($WantedRole -eq 'up') {
        $fusedRow = [int][math]::Floor($Native/2048)
        $column = [int]($Native % 2048)
        if ($Abi -eq 1) {
            if ($fusedRow -lt 768 -or $fusedRow -ge 1536) { return $null }
            return [pscustomobject]@{Row=$fusedRow-768;Column=$column}
        }
        if ($fusedRow -lt 0 -or $fusedRow -ge 768) { return $null }
        return [pscustomobject]@{Row=$fusedRow;Column=$column}
    }
    $column = [int][math]::Floor($Native/768)
    $row = [int]($Native % 768)
    if ($column -lt 0 -or $column -ge 2048) { return $null }
    return [pscustomobject]@{Row=$row;Column=$column}
}

$allV1Stage0Lines = [System.Collections.Generic.List[string]]::new()
$abi1OriginalLines = [System.Collections.Generic.List[string]]::new()
$abi1Bundles = [System.Collections.Generic.List[object]]::new()
$globalBySplit = @{fit=(HashSet-Ordinal);score=(HashSet-Ordinal)}
$attempts = [int64]0
$attemptsByAbi = @([int64]0,[int64]0,[int64]0)
for ($abi=0; $abi -lt 3; $abi++) {
    $abiId = [string]$family.abis[$abi].abi_id
    $usedByIdentity = @{}
    foreach ($identity in $identities) {
        $identityKey = "$abi|$($identity.Expert)|$($identity.Role)"
        $usedByIdentity[$identityKey] = HashSet-Ordinal
        $coordinateCount = if ($highCount.Contains("$($identity.Expert)|$($identity.Role)")) {24} else {20}
        foreach ($split in @('fit','score')) {
            $bundleTarget = [int]($coordinateCount/4)
            $accepted = 0
            [uint64]$counter = 0
            $nCall = if ($identity.Role -eq 'up') {[int64]$family.abis[$abi].up_numel} else {[int64]$family.abis[$abi].down_numel}
            [int64]$validBundleCount = [int64][math]::Floor(($nCall-1-3*$stride)/(4*$stride))+1
            while ($accepted -lt $bundleTarget) {
                Assert-True ($counter -lt [uint64]1000000) "bundle search terminates abi=$abi"
                $hash = Hash-Text "FUSEED-U32-v1|$abiId|$($identity.Expert)|$($identity.Role)|$split|$counter"
                [int64]$sequence = [int64]([uint64](U32-LE $hash 0) % [uint64]$stride)
                [int64]$normal4Index = [int64]([uint64](U32-LE $hash 4) % [uint64]$validBundleCount)
                $coordinates = [System.Collections.Generic.List[object]]::new()
                $localKeys = HashSet-Ordinal
                $valid = $true
                for ($lane=0; $lane -lt 4; $lane++) {
                    [int64]$native = $sequence+$stride*(4*$normal4Index+$lane)
                    if ($native -lt 0 -or $native -ge $nCall) { $valid=$false; break }
                    $coordinate = Invert-Native $abi $identity.Expert $identity.Role $native
                    if ($null -eq $coordinate) { $valid=$false; break }
                    $key = Canonical-Key $identity.Expert $identity.Role $coordinate.Row $coordinate.Column
                    if (-not $localKeys.Add($key)) { $valid=$false; break }
                    if ($usedByIdentity[$identityKey].Contains($key)) { $valid=$false; break }
                    $opposite = if ($split -eq 'fit') {'score'} else {'fit'}
                    if ($globalBySplit[$opposite].Contains($key)) { $valid=$false; break }
                    $coordinates.Add([pscustomobject]@{
                        Lane=[int]$lane;Row=[int]$coordinate.Row;Column=[int]$coordinate.Column;
                        Native=[int64]$native;Key=[string]$key
                    })
                }
                $counter++
                $attempts++
                $attemptsByAbi[$abi]++
                if (-not $valid) { continue }
                foreach ($coordinate in $coordinates) {
                    [void]$usedByIdentity[$identityKey].Add($coordinate.Key)
                    [void]$globalBySplit[$split].Add($coordinate.Key)
                    $line = "stage0|abi=$abi|$split|e=$($identity.Expert)|role=$($identity.Role)|r=$($coordinate.Row)|c=$($coordinate.Column)|seq=$sequence|j=$normal4Index|lane=$($coordinate.Lane)|native=$($coordinate.Native)"
                    $allV1Stage0Lines.Add($line)
                    if ($abi -eq 1) { $abi1OriginalLines.Add($line) }
                }
                if ($abi -eq 1) {
                    $category = "$($identity.Role)_$split"
                    $localExpert = [int]($identity.Expert % 32)
                    $seedDelta = [int](1024 + 100*[math]::Floor($identity.Expert/32))
                    $initializerOffset = if ($identity.Role -eq 'up') {11520+16*$localExpert} else {12032+8*$localExpert}
                    $scaleBits = if ($identity.Role -eq 'up') {'3c03126f'} else {'3a560a28'}
                    $abi1Bundles.Add([pscustomobject]@{
                        Category=[string]$category;Expert=[int]$identity.Expert;Role=[string]$identity.Role;
                        Split=[string]$split;AcceptedIndex=[int]$accepted;Sequence=[int64]$sequence;
                        Normal4Index=[int64]$normal4Index;SeedDelta=[int]$seedDelta;
                        InitializerOffset=[int]$initializerOffset;ScaleBits=[string]$scaleBits;
                        Coordinates=@($coordinates)
                    })
                }
                $accepted++
            }
        }
    }
}
Assert-Eq $allV1Stage0Lines.Count 3072 'reconstructed v1 stage0 record count'
Assert-Eq $abi1OriginalLines.Count 1024 'ABI1 subset coordinate count'
Assert-Eq $abi1Bundles.Count 256 'ABI1 subset bundle count'
$v1Stage0Digest = Digest-Lines $allV1Stage0Lines
Assert-Eq $v1Stage0Digest ([string]$v1.coordinate_protocol.expected_stage0_plan_digest_sha256) 'reconstructed v1 stage0 digest'
Assert-Eq $attempts ([int64]$v1.coordinate_protocol.expected_bundle_search_attempts) 'reconstructed v1 bundle attempts'

function Fill-Plan([string]$Namespace, [string]$Split, [System.Collections.Generic.HashSet[string]]$Set, [System.Collections.Generic.HashSet[string]]$Opposite, [object[]]$Ids, [int]$Target) {
    [uint64]$counter=0
    while ($Set.Count -lt $Target) {
        Assert-True ($counter -lt [uint64]10000000) "plan fill terminates namespace=$Namespace split=$Split"
        $hash = Hash-Text "FUSEED-U32-v1|$Namespace|$Split|$counter"
        $index = [int]([uint64](U32-LE $hash 0) % [uint64]$Ids.Count)
        $row = [int]([uint64](U32-LE $hash 4) % [uint64]768)
        $column = [int]([uint64](U32-LE $hash 8) % [uint64]2048)
        $identity = $Ids[$index]
        $key = Canonical-Key $identity.Expert $identity.Role $row $column
        if (-not $Opposite.Contains($key)) { [void]$Set.Add($key) }
        $counter++
    }
}

$stage1Fit = HashSet-Ordinal; $stage1Score = HashSet-Ordinal
foreach ($key in $globalBySplit.fit) { [void]$stage1Fit.Add($key) }
foreach ($key in $globalBySplit.score) { [void]$stage1Score.Add($key) }
Fill-Plan 'stage1' 'fit' $stage1Fit $stage1Score @($identities) 2048
Fill-Plan 'stage1' 'score' $stage1Score $stage1Fit @($identities) 2048
$fullFit = HashSet-Ordinal; $fullScore = HashSet-Ordinal
foreach ($key in $stage1Fit) { [void]$fullFit.Add($key) }
foreach ($key in $stage1Score) { [void]$fullScore.Add($key) }
Fill-Plan 'stage2' 'fit' $fullFit $fullScore @($identities) 24312
Fill-Plan 'stage2' 'score' $fullScore $fullFit @($identities) 24312
$validationFit = HashSet-Ordinal; $validationScore = HashSet-Ordinal
Fill-Plan 'validation' 'fit' $validationFit $validationScore @($validationIdentities) 8456
Fill-Plan 'validation' 'score' $validationScore $validationFit @($validationIdentities) 8456

$stage1FitSorted=@($stage1Fit); [Array]::Sort($stage1FitSorted,[StringComparer]::Ordinal)
$stage1ScoreSorted=@($stage1Score); [Array]::Sort($stage1ScoreSorted,[StringComparer]::Ordinal)
$fullFitSorted=@($fullFit); [Array]::Sort($fullFitSorted,[StringComparer]::Ordinal)
$fullScoreSorted=@($fullScore); [Array]::Sort($fullScoreSorted,[StringComparer]::Ordinal)
$validationFitSorted=@($validationFit); [Array]::Sort($validationFitSorted,[StringComparer]::Ordinal)
$validationScoreSorted=@($validationScore); [Array]::Sort($validationScoreSorted,[StringComparer]::Ordinal)

# Rebuild the complete parent plan digest as an anti-drift proof.
$allV1PlanLines = [System.Collections.Generic.List[string]]::new()
foreach ($line in $allV1Stage0Lines) { $allV1PlanLines.Add($line) }
foreach ($key in $stage1FitSorted) { $allV1PlanLines.Add("stage1|fit|$key") }
foreach ($key in $stage1ScoreSorted) { $allV1PlanLines.Add("stage1|score|$key") }
foreach ($key in $fullFitSorted) { $allV1PlanLines.Add("stage2|fit|$key") }
foreach ($key in $fullScoreSorted) { $allV1PlanLines.Add("stage2|score|$key") }
foreach ($key in $validationFitSorted) { $allV1PlanLines.Add("validation|fit|$key") }
foreach ($key in $validationScoreSorted) { $allV1PlanLines.Add("validation|score|$key") }
Assert-Eq (Digest-Lines $allV1PlanLines) ([string]$v1.coordinate_protocol.expected_plan_digest_sha256) 'reconstructed parent full plan digest'

# Freeze an execution order independent of any payload statistic: category,
# expert, accepted-bundle index. The lane order remains 0,1,2,3.
$categoryIndex = @{up_fit=0;down_fit=1;up_score=2;down_score=3}
$orderedBundles = @($abi1Bundles | Sort-Object @{Expression={$categoryIndex[$_.Category]}},Expert,AcceptedIndex)
$categoryBundleCounts = @{up_fit=0;down_fit=0;up_score=0;down_score=0}
$abi1FitKeys = HashSet-Ordinal
$abi1ScoreKeys = HashSet-Ordinal
$bundleLines = [System.Collections.Generic.List[string]]::new()
for ($index=0; $index -lt $orderedBundles.Count; $index++) {
    $bundle = $orderedBundles[$index]
    Assert-True ($categoryIndex.ContainsKey([string]$bundle.Category)) "known category at bundle $index"
    $categoryBundleCounts[$bundle.Category]++
    if ($index -gt 0) {
        Assert-True ($categoryIndex[$orderedBundles[$index-1].Category] -le $categoryIndex[$bundle.Category]) "category monotonicity at bundle $index"
    }
    Assert-Eq @($bundle.Coordinates).Count 4 "four coordinates at bundle $index"
    $expectedDelta = 1024 + 100*[math]::Floor($bundle.Expert/32)
    $expectedOffset = if ($bundle.Role -eq 'up') {11520+16*($bundle.Expert%32)} else {12032+8*($bundle.Expert%32)}
    Assert-Eq ([int]$bundle.SeedDelta) ([int]$expectedDelta) "seed delta at bundle $index"
    Assert-Eq ([int]$bundle.InitializerOffset) ([int]$expectedOffset) "initializer offset at bundle $index"
    $line = 'bundle|index={0:D3}|category={1}|e={2:D3}|role={3}|split={4}|accepted={5:D2}|seed_delta={6}|offset={7}|scale={8}|seq={9}|j={10}' -f `
        $index,$bundle.Category,$bundle.Expert,$bundle.Role,$bundle.Split,$bundle.AcceptedIndex,$bundle.SeedDelta,$bundle.InitializerOffset,$bundle.ScaleBits,$bundle.Sequence,$bundle.Normal4Index
    foreach ($coordinate in $bundle.Coordinates) {
        Assert-Eq ([int64]$coordinate.Native) ([int64]$bundle.Sequence+$stride*(4*[int64]$bundle.Normal4Index+[int64]$coordinate.Lane)) "native inverse at bundle $index lane $($coordinate.Lane)"
        Assert-Eq ([int]$coordinate.Lane) ([int]([math]::Floor($coordinate.Native/$stride) -band 3)) "lane inverse at bundle $index"
        Assert-Eq ([int64]$bundle.Normal4Index) ([int64]([math]::Floor($coordinate.Native/$stride) -shr 2)) "normal4 inverse at bundle $index"
        if ($bundle.Split -eq 'fit') { $targetSet=$abi1FitKeys } else { $targetSet=$abi1ScoreKeys }
        Assert-True ($targetSet.Add([string]$coordinate.Key)) "unique ABI1 coordinate at bundle $index lane $($coordinate.Lane)"
        $line += '|lane{0}=r{1:D3},c{2:D4},native{3}' -f $coordinate.Lane,$coordinate.Row,$coordinate.Column,$coordinate.Native
    }
    $bundleLines.Add($line)
}
Assert-Eq $categoryBundleCounts.up_fit 61 'up-fit bundle count'
Assert-Eq $categoryBundleCounts.down_fit 67 'down-fit bundle count'
Assert-Eq $categoryBundleCounts.up_score 61 'up-score bundle count'
Assert-Eq $categoryBundleCounts.down_score 67 'down-score bundle count'
Assert-Eq $abi1FitKeys.Count 512 'ABI1 fit coordinate count'
Assert-Eq $abi1ScoreKeys.Count 512 'ABI1 score coordinate count'
foreach ($key in $abi1FitKeys) {
    Assert-True (-not $abi1ScoreKeys.Contains($key)) "ABI1 fit/score disjoint: $key"
    Assert-True ($stage1Fit.Contains($key)) "ABI1 fit nested in stage1: $key"
}
foreach ($key in $abi1ScoreKeys) {
    Assert-True ($stage1Score.Contains($key)) "ABI1 score nested in stage1: $key"
}
foreach ($key in $stage1Fit) {
    Assert-True ($fullFit.Contains($key)) "stage1 fit nested in full: $key"
    Assert-True (-not $stage1Score.Contains($key)) "stage1 split disjoint: $key"
}
foreach ($key in $stage1Score) { Assert-True ($fullScore.Contains($key)) "stage1 score nested in full: $key" }
foreach ($key in $fullFit) { Assert-True (-not $fullScore.Contains($key)) "full split disjoint: $key" }
foreach ($key in $validationFit) { Assert-True (-not $validationScore.Contains($key)) "validation split disjoint: $key" }

$facts = [ordered]@{
    parent_v1_stage0_sha256=$v1Stage0Digest
    parent_v1_full_plan_sha256=(Digest-Lines $allV1PlanLines)
    abi1_original_stage0_records_sha256=(Digest-Lines $abi1OriginalLines)
    abi1_category_ordered_bundle_array_sha256=(Digest-Lines $bundleLines)
    stage1_fit_set_sha256=(Digest-Lines @($stage1FitSorted | ForEach-Object {"stage1|fit|$_"}))
    stage1_score_set_sha256=(Digest-Lines @($stage1ScoreSorted | ForEach-Object {"stage1|score|$_"}))
    full_fit_set_sha256=(Digest-Lines @($fullFitSorted | ForEach-Object {"stage2|fit|$_"}))
    full_score_set_sha256=(Digest-Lines @($fullScoreSorted | ForEach-Object {"stage2|score|$_"}))
    validation_fit_set_sha256=(Digest-Lines @($validationFitSorted | ForEach-Object {"validation|fit|$_"}))
    validation_score_set_sha256=(Digest-Lines @($validationScoreSorted | ForEach-Object {"validation|score|$_"}))
    abi1_stage0_coordinate_records=$abi1OriginalLines.Count
    abi1_stage0_bundles=$orderedBundles.Count
    abi1_stage0_attempts=$attemptsByAbi[1]
    stage1_fit=$stage1Fit.Count
    stage1_score=$stage1Score.Count
    full_fit=$fullFit.Count
    full_score=$fullScore.Count
    validation_fit=$validationFit.Count
    validation_score=$validationScore.Count
}
$v2PlanLines = [System.Collections.Generic.List[string]]::new()
$v2PlanLines.Add('family|FUSEED-PMG1-v2')
$v2PlanLines.Add('abi|CURRENT_PMG_GATE_UP_DIRECT_BF16')
foreach ($line in $bundleLines) { $v2PlanLines.Add($line) }
foreach ($key in $stage1FitSorted) { $v2PlanLines.Add("stage1|fit|$key") }
foreach ($key in $stage1ScoreSorted) { $v2PlanLines.Add("stage1|score|$key") }
foreach ($key in $fullFitSorted) { $v2PlanLines.Add("stage2|fit|$key") }
foreach ($key in $fullScoreSorted) { $v2PlanLines.Add("stage2|score|$key") }
foreach ($key in $validationFitSorted) { $v2PlanLines.Add("validation|fit|$key") }
foreach ($key in $validationScoreSorted) { $v2PlanLines.Add("validation|score|$key") }
$facts.v2_complete_plan_sha256 = Digest-Lines $v2PlanLines

if ($PrintPlanFacts) { $facts | ConvertTo-Json -Depth 10 }
if ($PrintPlanJson) {
    [ordered]@{
        schema='fuseed_pmg1_v2_emitted_plan_v1'
        facts=$facts
        category_order=@('up_fit','down_fit','up_score','down_score')
        stage0_bundles=$orderedBundles
        stage1_fit=$stage1FitSorted
        stage1_score=$stage1ScoreSorted
        full_fit=$fullFitSorted
        full_score=$fullScoreSorted
        validation_fit=$validationFitSorted
        validation_score=$validationScoreSorted
    } | ConvertTo-Json -Depth 20
}

$lockedFacts = $design.coordinate_protocol.expected_facts
foreach ($property in $facts.Keys) {
    Assert-Eq ([string]$lockedFacts.$property) ([string]$facts[$property]) "locked plan fact $property"
}
foreach ($property in @('parent_v1_stage0_sha256','parent_v1_full_plan_sha256','abi1_original_stage0_records_sha256','abi1_category_ordered_bundle_array_sha256','stage1_fit_set_sha256','stage1_score_set_sha256','full_fit_set_sha256','full_score_set_sha256','validation_fit_set_sha256','validation_score_set_sha256','v2_complete_plan_sha256')) {
    Assert-Eq ([string]$receipt.plan_receipt.$property) ([string]$facts[$property]) "receipt plan fact $property"
}

# Candidate/Top-K/value ledgers.
$stage0Values = [int64]4294967296 * 1024
$stage0Cores = [int64]4294967296 * 256
$stage1Values = [int64]8192 * 4096
$stage2Values = [int64]256 * 48624
$validationValues = [int64]16912
Assert-Eq ([int64]$design.cascade.stage0.generated_normal_values) $stage0Values 'stage0 values'
Assert-Eq ([int64]$design.cascade.stage0.generated_counter_cores) $stage0Cores 'stage0 cores'
Assert-Eq ([int64]$design.cascade.stage1.generated_normal_values_max) $stage1Values 'stage1 values'
Assert-Eq ([int64]$design.cascade.stage2.generated_normal_values_max) $stage2Values 'stage2 values'
Assert-Eq ([int64]$design.cascade.validation.generated_normal_values) $validationValues 'validation values'
Assert-Eq ([int64]$design.cascade.maximum_generated_normal_values_total) ($stage0Values+$stage1Values+$stage2Values+$validationValues) 'total values'
Assert-Eq ([int64]$design.cascade.stage0.q_buffer_bytes) ([int64]16777216*8) 'one-shard binary64 metric buffer'
Assert-Eq ([int64]$design.cascade.stage0.all_shard_record_bytes) ([int64]256*8192*12) 'packed shard record ledger'
Assert-Eq ([int64]$design.cascade.stage0.rolling_topk_bytes) ([int64]8192*12) 'rolling Top-K bytes'

# Physical rate/read arithmetic. All serialized bytes are charged.
$targetWeights = [double]28311552
$weightsPerExpert = [double]4718592
$metadataBytes = [double]80
$sideBpw = $metadataBytes*8.0/$targetWeights
Assert-Close ([double]$design.physical_ledger.side_bpw) $sideBpw 1e-18 'side bpw'
Assert-Close ([double]$design.physical_ledger.maximum_base_codec_bpw_at_2_15) (2.15-$sideBpw) 1e-15 'base cap at 2.15'
Assert-Close ([double]$design.physical_ledger.maximum_base_codec_bpw_at_2_5) (2.5-$sideBpw) 1e-15 'base cap at 2.5'
foreach ($rate in @(2.15,2.5)) {
    $bytes = $weightsPerExpert*$rate/8.0
    $amplification = 1.169444+20.0/$bytes
    $row = @($design.physical_ledger.read_points | Where-Object {[double]$_.total_bpw -eq $rate})
    Assert-Eq $row.Count 1 "read row at $rate"
    Assert-Close ([double]$row[0].serialized_bytes_per_expert) $bytes 1e-8 "expert bytes at $rate"
    Assert-Close ([double]$row[0].appended_cold_read_amplification) $amplification 1e-15 "read amplification at $rate"
    Assert-True ($amplification -lt 2.0) "read cap at $rate"
}

# Red-team boundary: post-selection controls are not a multiplicity correction.
Assert-Eq ([bool]$design.scientific_claim.control_searches_seed_family) $false 'controls do not search seeds'
Assert-Eq ([bool]$design.scientific_claim.randomization_or_familywise_p_value_claimed) $false 'no control p-value claim'
Assert-Eq ([bool]$design.scientific_claim.exactly_one_validation_descriptor) $true 'one validation descriptor sentinel'
Assert-Eq ([bool]$design.scientific_claim.control_failure_permits_retry) $false 'no retry after control failure'
Assert-Eq ([bool]$design.scientific_claim.validation_failure_permits_retry) $false 'no retry after validation failure'

# Exact local artifact closure. The manifest deliberately excludes its own hash.
$manifestPath = Join-Path $root 'ARTIFACT_SHA256SUMS.txt'
Assert-True (Test-Path -LiteralPath $manifestPath -PathType Leaf) 'artifact manifest exists'
$expectedMembers = @('ARTIFACT_SHA256SUMS.txt','DESIGN_RECEIPT.json','README.md','RED_TEAM.md','design_lock.json','source_bindings.json','verify_design.ps1')
$actualMembers = @(Get-ChildItem -LiteralPath $root -Force)
Assert-Eq $actualMembers.Count $expectedMembers.Count 'exact package member count'
foreach ($member in $actualMembers) {
    Assert-True (-not ($member.Attributes -band [IO.FileAttributes]::ReparsePoint)) "package member is not a link: $($member.Name)"
    Assert-True (-not $member.PSIsContainer) "package member is a regular file: $($member.Name)"
    Assert-True ($expectedMembers -ccontains $member.Name) "expected package member: $($member.Name)"
}
$manifestRows = @(Get-Content -LiteralPath $manifestPath | Where-Object {$_.Length -gt 0})
Assert-Eq $manifestRows.Count ($expectedMembers.Count-1) 'manifest row count'
$seenManifest = HashSet-Ordinal
foreach ($row in $manifestRows) {
    Assert-True ($row -cmatch '^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$') "manifest row grammar: $row"
    $hash = [string]$Matches[1]
    $name = [string]$Matches[2]
    Assert-True ($name -cne 'ARTIFACT_SHA256SUMS.txt') 'manifest excludes itself'
    Assert-True ($expectedMembers -ccontains $name) "manifest member expected: $name"
    Assert-True ($seenManifest.Add($name)) "manifest member unique: $name"
    Assert-Eq (File-Sha256 (Join-Path $root $name)) $hash "manifest hash: $name"
}
Assert-Eq $seenManifest.Count ($expectedMembers.Count-1) 'manifest covers every non-manifest member'

[ordered]@{
    status='PASS_SOURCE_ONLY_DESIGN_DRAFT'
    checks=$script:Checks
    plan=$facts
    authorization='NONE'
} | ConvertTo-Json -Depth 10
