[CmdletBinding()]
param(
    [string]$Package = $PSScriptRoot,
    [switch]$SkipReceiptAndManifest,
    [switch]$PrintPlanFacts
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

Assert-True ([BitConverter]::IsLittleEndian) 'verifier host is little-endian'
$root = [System.IO.Path]::GetFullPath($Package)
Assert-True (Test-Path -LiteralPath $root -PathType Container) 'package exists'
$designPath = Join-Path $root 'design_lock.json'
$sourcesPath = Join-Path $root 'source_bindings.json'
Assert-True (Test-Path -LiteralPath $designPath -PathType Leaf) 'design lock exists'
Assert-True (Test-Path -LiteralPath $sourcesPath -PathType Leaf) 'source bindings exist'
$design = Get-Content -LiteralPath $designPath -Raw | ConvertFrom-Json -Depth 100
$sources = Get-Content -LiteralPath $sourcesPath -Raw | ConvertFrom-Json -Depth 100

Assert-Eq $design.schema 'fuseed_u32_source_only_design_v1' 'design schema'
Assert-Eq $design.status 'FROZEN_SOURCE_ONLY_DESIGN_EARLY_KILL_RUNTIME_NO_QWEN_PENDING_INDEPENDENT_DIRECT_CALIBRATION_AUDIT' 'design status'
Assert-True ([bool]$design.sealed) 'design sealed'
Assert-Eq $sources.schema 'fuseed_u32_source_bindings_v1' 'source schema'
Assert-Eq $sources.status 'SOURCE_ONLY_DESIGN_WITH_EXTERNAL_SYNTHETIC_RUNTIME_EVIDENCE_NO_MODEL_ACCESS' 'source status'

$family = $design.frozen_family
$n = [int64]4294967296
$abiCount = [int64]@($family.abis).Count
Assert-Eq ([int64]$family.base_seed_min_inclusive) 0 'base seed minimum'
Assert-Eq ([uint64]$family.base_seed_max_inclusive) ([uint64]4294967295) 'base seed maximum'
Assert-Eq ([int64]$family.base_seed_count) $n 'base seed count'
Assert-Eq $abiCount 3 'ABI count'
Assert-Eq ([int64]$family.abi_count) $abiCount 'locked ABI count'
Assert-Eq ([int64]$family.candidate_count) ($n*$abiCount) 'candidate count'
Assert-Eq ([int64]$family.candidate_count) 12884901888 'candidate literal'
Assert-Eq ([bool]$family.seed_zero_public_cli_provenance) $false 'seed zero provenance boundary'

$expectedAbi = @(
    @{Id='PAI6BA_LGP_GATE_UP_DIRECT_BF16'; Up=[int64]100663296; Down=[int64]50331648},
    @{Id='CURRENT_PMG_GATE_UP_DIRECT_BF16'; Up=[int64]3145728; Down=[int64]1572864},
    @{Id='CURRENT_PMG_UP_GATE_DIRECT_BF16'; Up=[int64]3145728; Down=[int64]1572864}
)
for ($i=0; $i -lt $expectedAbi.Count; $i++) {
    $row = $family.abis[$i]
    Assert-Eq ([int]$row.abi_index) $i "ABI index $i"
    Assert-Eq ([string]$row.abi_id) $expectedAbi[$i].Id "ABI id $i"
    Assert-Eq ([int64]$row.up_numel) $expectedAbi[$i].Up "ABI up numel $i"
    Assert-Eq ([int64]$row.down_numel) $expectedAbi[$i].Down "ABI down numel $i"
}
Assert-Eq @($family.blocked_abis).Count 4 'blocked ABI family count'

# Prove exact complete-u32 shard partition without allocating a seed vector.
$shardSize = [int64]16777216
$shardsPerAbi = [int64]256
$lastEnd = [int64]0
for ($s=0; $s -lt $shardsPerAbi; $s++) {
    $base = [int64]$s*$shardSize
    $end = $base+$shardSize
    Assert-Eq $base $lastEnd "contiguous seed shard $s"
    Assert-True ($base -ge 0 -and $end -le $n) "seed shard bounded $s"
    Assert-Eq ($end-$base) $shardSize "seed shard size $s"
    $lastEnd = $end
}
Assert-Eq $lastEnd $n 'seed shards cover all u32 labels'

# Effective MCore seed addition is u64-with-carry, never u32 wrap.
$addends = @(1024,1124,1224,1324)
Assert-Eq @($design.coherent_seed_map.selection_seed_addends).Count $addends.Count 'seed addend count'
for ($i=0; $i -lt $addends.Count; $i++) {
    Assert-Eq ([int]$design.coherent_seed_map.selection_seed_addends[$i]) $addends[$i] "seed addend $i"
}
$bases = @([uint64]0,[uint64]1,[uint64]4294965971,[uint64]4294965972,[uint64]4294966071,[uint64]4294966072,[uint64]4294966171,[uint64]4294966172,[uint64]4294966271,[uint64]4294966272,[uint64]4294967294,[uint64]4294967295)
foreach ($b in $bases) {
    foreach ($d in $addends) {
        $sum = $b+[uint64]$d
        $lo = [uint32]($sum -band [uint64]4294967295)
        $hi = [uint32]($sum -shr 32)
        Assert-Eq ([uint64]$lo) ($sum % [uint64]4294967296) "effective seed low b=$b d=$d"
        Assert-Eq ([uint64]$hi) ([uint64][math]::Floor([double]$sum/4294967296.0)) "effective seed high b=$b d=$d"
    }
}
Assert-Eq ([uint32](([uint64]4294966272+1024) -shr 32)) ([uint32]1) 'first +1024 carry'
Assert-Eq ([uint32](([uint64]4294965972+1324) -shr 32)) ([uint32]1) 'first +1324 carry'

# Independent Philox4x32-10 zero KAT using exact integer operations.
function Philox-Zero-Kat {
    [uint32]$c0=0; [uint32]$c1=0; [uint32]$c2=0; [uint32]$c3=0
    [uint32]$k0=0; [uint32]$k1=0
    [uint64]$m0=[Convert]::ToUInt32('d2511f53',16)
    [uint64]$m1=[Convert]::ToUInt32('cd9e8d57',16)
    [uint32]$w0=[Convert]::ToUInt32('9e3779b9',16)
    [uint32]$w1=[Convert]::ToUInt32('bb67ae85',16)
    for ($round=0; $round -lt 10; $round++) {
        [uint64]$p0=$m0*[uint64]$c0
        [uint64]$p1=$m1*[uint64]$c2
        [uint32]$lo0=[uint32]($p0 -band [uint64]4294967295)
        [uint32]$hi0=[uint32]($p0 -shr 32)
        [uint32]$lo1=[uint32]($p1 -band [uint64]4294967295)
        [uint32]$hi1=[uint32]($p1 -shr 32)
        [uint32]$n0=$hi1 -bxor $c1 -bxor $k0
        [uint32]$n1=$lo1
        [uint32]$n2=$hi0 -bxor $c3 -bxor $k1
        [uint32]$n3=$lo0
        $c0=$n0; $c1=$n1; $c2=$n2; $c3=$n3
        $k0=[uint32](([uint64]$k0+[uint64]$w0) -band [uint64]4294967295)
        $k1=[uint32](([uint64]$k1+[uint64]$w1) -band [uint64]4294967295)
    }
    @($c0,$c1,$c2,$c3)
}
$kat = @(Philox-Zero-Kat)
$katWant = @('6627e8d5','e169c58d','bc57ac4c','9b00dbd8')
for ($i=0; $i -lt 4; $i++) {
    Assert-Eq ('{0:x8}' -f $kat[$i]) $katWant[$i] "Philox zero KAT word $i"
    Assert-Eq ([string]$design.philox_direct_counter.philox_constants.zero_kat_words[$i]) $katWant[$i] "locked KAT word $i"
}

# Validate counter quotient/sequence/lane arithmetic at exact boundaries.
$stride = [int64]261120
Assert-Eq ([int64]$design.philox_direct_counter.stride) $stride 'Philox stride'
$nativeTests = @([int64]0,[int64]1,($stride-1),$stride,($stride+1),(4*$stride-1),(4*$stride),(4*$stride+1),[int64]100663295,[int64]50331647,[int64]3145727,[int64]1572863)
foreach ($native in $nativeTests) {
    $seq = $native % $stride
    $q = [int64][math]::Floor($native/$stride)
    $lane = $q -band 3
    $j = $q -shr 2
    Assert-Eq ($seq+$stride*(4*$j+$lane)) $native "counter inverse native=$native"
    Assert-True ($seq -ge 0 -and $seq -lt $stride) "sequence range native=$native"
    Assert-True ($lane -ge 0 -and $lane -le 3) "lane range native=$native"
}

# Frozen selection identities and deterministic high-count quota.
$selectionExperts = @(0,8,16,32,40,48,64,72,80,96,104,112)
$validationExperts = @(24,56,88,120)
$identities = [System.Collections.Generic.List[object]]::new()
foreach ($e in $selectionExperts) {
    foreach ($role in @('up','down')) {
        if ($e -eq 0 -and $role -eq 'up') { continue }
        $identities.Add([pscustomobject]@{Expert=[int]$e;Role=$role})
    }
}
Assert-Eq $identities.Count 23 'selection identity count'
$validationIdentities = [System.Collections.Generic.List[object]]::new()
foreach ($e in $validationExperts) {
    foreach ($role in @('up','down')) { $validationIdentities.Add([pscustomobject]@{Expert=[int]$e;Role=$role}) }
}
Assert-Eq $validationIdentities.Count 8 'validation identity count'

$highCount = HashSet-Ordinal
foreach ($roleSpec in @(@{Role='up';Count=6},@{Role='down';Count=7})) {
    $ranked = @($identities | Where-Object Role -eq $roleSpec.Role | ForEach-Object {
        $text = "FUSEED-U32-v1|quota|$($_.Expert)|$($_.Role)"
        [pscustomobject]@{Key=(Canonical-Key $_.Expert $_.Role 0 0); Id=$_; Hash=(Text-Sha256 $text)}
    } | Sort-Object Hash,Key)
    Assert-True ($ranked.Count -ge $roleSpec.Count) "quota candidates $($roleSpec.Role)"
    for ($i=0; $i -lt $roleSpec.Count; $i++) {
        [void]$highCount.Add("$($ranked[$i].Id.Expert)|$($ranked[$i].Id.Role)")
    }
}
Assert-Eq $highCount.Count 13 'high-count identities'

function Invert-Native([int]$Abi, [int]$WantedExpert, [string]$WantedRole, [int64]$Native) {
    if ($Abi -eq 0) {
        if ($WantedRole -eq 'up') {
            [int64]$block = 2048*1536
            $le = [int][math]::Floor($Native/$block)
            $within = $Native % $block
            $c = [int][math]::Floor($within/1536)
            $fusedRow = [int]($within % 1536)
            if ($le -ne ($WantedExpert % 32) -or $fusedRow -lt 768) { return $null }
            return [pscustomobject]@{Row=$fusedRow-768;Column=$c}
        }
        [int64]$block = 768*2048
        $le = [int][math]::Floor($Native/$block)
        $within = $Native % $block
        $r = [int][math]::Floor($within/2048)
        $c = [int]($within % 2048)
        if ($le -ne ($WantedExpert % 32)) { return $null }
        return [pscustomobject]@{Row=$r;Column=$c}
    }
    if ($WantedRole -eq 'up') {
        $fusedRow = [int][math]::Floor($Native/2048)
        $c = [int]($Native % 2048)
        if ($Abi -eq 1) {
            if ($fusedRow -lt 768 -or $fusedRow -ge 1536) { return $null }
            return [pscustomobject]@{Row=$fusedRow-768;Column=$c}
        }
        if ($fusedRow -lt 0 -or $fusedRow -ge 768) { return $null }
        return [pscustomobject]@{Row=$fusedRow;Column=$c}
    }
    $c = [int][math]::Floor($Native/768)
    $r = [int]($Native % 768)
    if ($c -lt 0 -or $c -ge 2048) { return $null }
    return [pscustomobject]@{Row=$r;Column=$c}
}

# Enumerate the exact source-independent four-lane bundle plan.
$stage0Lines = [System.Collections.Generic.List[string]]::new()
$allPlanLines = [System.Collections.Generic.List[string]]::new()
$globalBySplit = @{fit=(HashSet-Ordinal);score=(HashSet-Ordinal)}
$attempts = [int64]0
for ($abi=0; $abi -lt 3; $abi++) {
    $abiId = [string]$family.abis[$abi].abi_id
    $usedByIdentity = @{}
    foreach ($id in $identities) {
        $identityKey = "$abi|$($id.Expert)|$($id.Role)"
        $usedByIdentity[$identityKey] = HashSet-Ordinal
        $coordinateCount = if ($highCount.Contains("$($id.Expert)|$($id.Role)")) {24} else {20}
        foreach ($split in @('fit','score')) {
            $bundleTarget = [int]($coordinateCount/4)
            $accepted = 0
            [uint64]$counter = 0
            $nCall = if ($id.Role -eq 'up') {[int64]$family.abis[$abi].up_numel} else {[int64]$family.abis[$abi].down_numel}
            [int64]$validBundleCount = [int64][math]::Floor(($nCall-1-3*$stride)/(4*$stride))+1
            Assert-True ($validBundleCount -gt 0) "valid bundle count abi=$abi role=$($id.Role)"
            while ($accepted -lt $bundleTarget) {
                Assert-True ($counter -lt [uint64]1000000) "bundle search terminates abi=$abi expert=$($id.Expert) role=$($id.Role) split=$split"
                $hash = Hash-Text "FUSEED-U32-v1|$abiId|$($id.Expert)|$($id.Role)|$split|$counter"
                [int64]$seq = [int64]([uint64](U32-LE $hash 0) % [uint64]$stride)
                [int64]$j = [int64]([uint64](U32-LE $hash 4) % [uint64]$validBundleCount)
                $coords = [System.Collections.Generic.List[object]]::new()
                $localKeys = HashSet-Ordinal
                $valid = $true
                for ($lane=0; $lane -lt 4; $lane++) {
                    [int64]$native = $seq+$stride*(4*$j+$lane)
                    if ($native -lt 0 -or $native -ge $nCall) { $valid=$false; break }
                    $coord = Invert-Native $abi $id.Expert $id.Role $native
                    if ($null -eq $coord) { $valid=$false; break }
                    $key = Canonical-Key $id.Expert $id.Role $coord.Row $coord.Column
                    if (-not $localKeys.Add($key)) { $valid=$false; break }
                    if ($usedByIdentity[$identityKey].Contains($key)) { $valid=$false; break }
                    $opposite = if ($split -eq 'fit') {'score'} else {'fit'}
                    if ($globalBySplit[$opposite].Contains($key)) { $valid=$false; break }
                    $coords.Add([pscustomobject]@{Row=$coord.Row;Column=$coord.Column;Native=$native;Key=$key;Lane=$lane})
                }
                $counter++
                $attempts++
                if (-not $valid) { continue }
                foreach ($coord in $coords) {
                    [void]$usedByIdentity[$identityKey].Add($coord.Key)
                    [void]$globalBySplit[$split].Add($coord.Key)
                    $line = "stage0|abi=$abi|$split|e=$($id.Expert)|role=$($id.Role)|r=$($coord.Row)|c=$($coord.Column)|seq=$seq|j=$j|lane=$($coord.Lane)|native=$($coord.Native)"
                    $stage0Lines.Add($line)
                    $allPlanLines.Add($line)
                    Assert-Eq ($coord.Native % $stride) $seq "bundle sequence abi=$abi native=$($coord.Native)"
                    $q = [int64][math]::Floor($coord.Native/$stride)
                    Assert-Eq ($q -band 3) $coord.Lane "bundle lane abi=$abi native=$($coord.Native)"
                    Assert-Eq ($q -shr 2) $j "bundle normal4 index abi=$abi native=$($coord.Native)"
                    Assert-True ($coord.Row -ge 0 -and $coord.Row -lt 768) "bundle row range"
                    Assert-True ($coord.Column -ge 0 -and $coord.Column -lt 2048) "bundle column range"
                }
                $accepted++
            }
            Assert-Eq ($accepted*4) $coordinateCount "matrix split coordinate quota abi=$abi expert=$($id.Expert) role=$($id.Role) split=$split"
        }
    }
}
Assert-Eq $stage0Lines.Count (3*1024) 'all ABI stage0 coordinate records'

function Fill-Plan([string]$Namespace, [string]$Split, [System.Collections.Generic.HashSet[string]]$Set, [System.Collections.Generic.HashSet[string]]$Opposite, [object[]]$Ids, [int]$Target) {
    [uint64]$counter=0
    while ($Set.Count -lt $Target) {
        Assert-True ($counter -lt [uint64]10000000) "plan fill terminates namespace=$Namespace split=$Split"
        $hash = Hash-Text "FUSEED-U32-v1|$Namespace|$Split|$counter"
        $index = [int]([uint64](U32-LE $hash 0) % [uint64]$Ids.Count)
        $row = [int]([uint64](U32-LE $hash 4) % [uint64]768)
        $column = [int]([uint64](U32-LE $hash 8) % [uint64]2048)
        $id = $Ids[$index]
        $key = Canonical-Key $id.Expert $id.Role $row $column
        if (-not $Opposite.Contains($key)) { [void]$Set.Add($key) }
        $counter++
    }
    Assert-Eq $Set.Count $Target "plan target namespace=$Namespace split=$Split"
}

# Common stage1 contains every ABI-specific stage0 coordinate.
$stage1Fit = HashSet-Ordinal
$stage1Score = HashSet-Ordinal
foreach ($key in $globalBySplit.fit) { [void]$stage1Fit.Add($key) }
foreach ($key in $globalBySplit.score) { [void]$stage1Score.Add($key) }
foreach ($key in $stage1Fit) { Assert-True (-not $stage1Score.Contains($key)) "stage0 fit/score disjoint $key" }
Fill-Plan 'stage1' 'fit' $stage1Fit $stage1Score @($identities) 2048
Fill-Plan 'stage1' 'score' $stage1Score $stage1Fit @($identities) 2048
$stage1FitSorted = @($stage1Fit); [Array]::Sort($stage1FitSorted,[StringComparer]::Ordinal)
$stage1ScoreSorted = @($stage1Score); [Array]::Sort($stage1ScoreSorted,[StringComparer]::Ordinal)
foreach ($key in $stage1FitSorted) { $allPlanLines.Add("stage1|fit|$key") }
foreach ($key in $stage1ScoreSorted) { $allPlanLines.Add("stage1|score|$key") }

# Full selection is a strict nested extension of stage1.
$fullFit = HashSet-Ordinal
$fullScore = HashSet-Ordinal
foreach ($key in $stage1Fit) { [void]$fullFit.Add($key) }
foreach ($key in $stage1Score) { [void]$fullScore.Add($key) }
Fill-Plan 'stage2' 'fit' $fullFit $fullScore @($identities) 24312
Fill-Plan 'stage2' 'score' $fullScore $fullFit @($identities) 24312
foreach ($key in $stage1Fit) { Assert-True ($fullFit.Contains($key)) "stage1 fit nested $key" }
foreach ($key in $stage1Score) { Assert-True ($fullScore.Contains($key)) "stage1 score nested $key" }
foreach ($key in $fullFit) { Assert-True (-not $fullScore.Contains($key)) "full selection fit/score disjoint $key" }
$fullFitSorted = @($fullFit); [Array]::Sort($fullFitSorted,[StringComparer]::Ordinal)
$fullScoreSorted = @($fullScore); [Array]::Sort($fullScoreSorted,[StringComparer]::Ordinal)
foreach ($key in $fullFitSorted) { $allPlanLines.Add("stage2|fit|$key") }
foreach ($key in $fullScoreSorted) { $allPlanLines.Add("stage2|score|$key") }

# Validation is prospectively fixed on disjoint experts.
$valFit = HashSet-Ordinal
$valScore = HashSet-Ordinal
Fill-Plan 'validation' 'fit' $valFit $valScore @($validationIdentities) 8456
Fill-Plan 'validation' 'score' $valScore $valFit @($validationIdentities) 8456
foreach ($key in $valFit) { Assert-True (-not $valScore.Contains($key)) "validation fit/score disjoint $key" }
$valFitSorted = @($valFit); [Array]::Sort($valFitSorted,[StringComparer]::Ordinal)
$valScoreSorted = @($valScore); [Array]::Sort($valScoreSorted,[StringComparer]::Ordinal)
foreach ($key in $valFitSorted) { $allPlanLines.Add("validation|fit|$key") }
foreach ($key in $valScoreSorted) { $allPlanLines.Add("validation|score|$key") }

$stage0Digest = Text-Sha256 (($stage0Lines -join "`n")+"`n")
$planDigest = Text-Sha256 (($allPlanLines -join "`n")+"`n")
if ($PrintPlanFacts) {
    [pscustomobject]@{
        stage0_sha256=$stage0Digest
        all_plan_sha256=$planDigest
        stage0_records=$stage0Lines.Count
        stage0_unique_fit=$globalBySplit.fit.Count
        stage0_unique_score=$globalBySplit.score.Count
        stage1_fit=$stage1Fit.Count
        stage1_score=$stage1Score.Count
        full_fit=$fullFit.Count
        full_score=$fullScore.Count
        validation_fit=$valFit.Count
        validation_score=$valScore.Count
        bundle_attempts=$attempts
    } | ConvertTo-Json -Depth 5
}
Assert-Eq ([string]$design.coordinate_protocol.expected_plan_digest_sha256) $planDigest 'frozen coordinate plan digest'
Assert-Eq ([string]$design.coordinate_protocol.expected_stage0_plan_digest_sha256) $stage0Digest 'frozen stage0 plan digest'
Assert-Eq ([int64]$design.coordinate_protocol.expected_bundle_search_attempts) $attempts 'frozen bundle search attempts'

# Cascade arithmetic and bounded memory.
$cascade = $design.cascade
$stage0 = $cascade.stage0
$stage1 = $cascade.stage1
$stage2 = $cascade.stage2
$validation = $cascade.validation
$candidateCount = $n*$abiCount
$stage0Values = $candidateCount*[int64]1024
$stage0Cores = $candidateCount*[int64]256
Assert-Eq ([int64]$stage0.seed_shards_per_abi) 256 'stage0 shards per ABI'
Assert-Eq ([int64]$stage0.total_seed_shards) 768 'stage0 total shards'
Assert-Eq ([int64]$stage0.generated_normal_values) $stage0Values 'stage0 generated values'
Assert-Eq ([int64]$stage0.generated_philox4_cores) $stage0Cores 'stage0 Philox cores'
Assert-Eq ([int64]$stage0.candidate_domain_metrics) ($candidateCount*33) 'stage0 candidate-domain metrics'
Assert-Eq ([int64]$stage0.domain_cross_moment_fmas) ($candidateCount*33*1024) 'stage0 cross FMAs'
Assert-Eq ([int64]$stage0.anchor_square_fmas) $stage0Values 'stage0 square FMAs'
Assert-Eq ([int64]$design.fused_stage0_kernel.q_buffer_bytes) ([int64]33*16777216*4) 'q buffer bytes'
Assert-Eq ([int64]$stage0.topk_state_bytes_per_shard) ([int64]33*8192*8) 'topK state bytes per shard'
Assert-Eq ([int64]$stage0.topk_state_bytes_all_shards) ([int64]768*33*8192*8) 'all shard TopK state bytes'
$union1 = [int64]3*33*8192
Assert-Eq ([int64]$stage1.deduplicated_union_candidates_max) $union1 'stage1 union max'
Assert-Eq ([int64]$stage1.generated_normal_values_max) ($union1*4096) 'stage1 generated values'
$union2 = [int64]33*256
Assert-Eq ([int64]$stage2.deduplicated_union_candidates_max) $union2 'stage2 union max'
Assert-Eq ([int64]$stage2.generated_normal_values_max) ($union2*48624) 'stage2 generated values'
Assert-Eq ([int64]$validation.generated_normal_values) ([int64]33*16912) 'validation values'
$totalValues = $stage0Values+[int64]$stage1.generated_normal_values_max+[int64]$stage2.generated_normal_values_max+[int64]$validation.generated_normal_values
Assert-Eq ([int64]$cascade.maximum_generated_normal_values_total) $totalValues 'total generated values'
$cio=$design.stage0_compute_io_ledger
Assert-Eq ([int64]$cio.candidate_domain_metrics) ($candidateCount*33) 'compute ledger metrics'
Assert-Eq ([int64]$cio.domain_cross_moment_fmas) ($candidateCount*33*1024) 'compute ledger cross FMAs'
Assert-Eq ([int64]$cio.anchor_square_fmas) $stage0Values 'compute ledger square FMAs'
Assert-Eq ([int64]$cio.fp16_alpha_mu_conversions_two_roles) ($candidateCount*33*4) 'compute ledger FP16 conversions'
Assert-Eq ([int64]$cio.logical_float32_target_coefficient_read_bytes) ($candidateCount*33*1024*4) 'compute ledger logical coefficient bytes'
Assert-Eq ([int64]$cio.avoided_float32_anchor_write_bytes) ($stage0Values*4) 'compute ledger avoided anchor bytes'
Assert-Eq ([int64]$cio.float32_metric_write_bytes) ($candidateCount*33*4) 'compute ledger metric bytes'

# Exhaustive small-universe proof tests for partitioned exact TopK and tie order.
function Top-K([object[]]$Rows, [int]$K) {
    @($Rows | Sort-Object @{Expression={$_.Metric};Descending=$true},@{Expression={$_.Abi};Ascending=$true},@{Expression={$_.Seed};Ascending=$true} | Select-Object -First $K)
}
for ($trial=0; $trial -lt 64; $trial++) {
    $rows = [System.Collections.Generic.List[object]]::new()
    for ($i=0; $i -lt 73; $i++) {
        $metric = [int](($i*17+$trial*11+($i%5)*13) % 19)
        $rows.Add([pscustomobject]@{Metric=$metric;Abi=($i%3);Seed=[uint32](($i*37+$trial*101)%257)})
    }
    foreach ($k in @(1,2,3,7,8,16)) {
        $full = @(Top-K @($rows) $k)
        $parts = [System.Collections.Generic.List[object]]::new()
        for ($p=0; $p -lt 7; $p++) {
            $part = @($rows | Where-Object { ([int]$_.Seed % 7) -eq $p })
            foreach ($row in @(Top-K $part $k)) { $parts.Add($row) }
        }
        $merged = @(Top-K @($parts) $k)
        Assert-Eq $merged.Count $full.Count "TopK merge count trial=$trial k=$k"
        for ($i=0; $i -lt $full.Count; $i++) {
            $a="$($full[$i].Metric)|$($full[$i].Abi)|$($full[$i].Seed)"
            $b="$($merged[$i].Metric)|$($merged[$i].Abi)|$($merged[$i].Seed)"
            Assert-Eq $b $a "TopK union theorem trial=$trial k=$k i=$i"
        }
    }
}

# Retention planning formula is descriptive only but its arithmetic is frozen.
$m = 512.0; $Nf=4294967296.0; $Kf=8192.0; $capture=0.1456888483858212
$sqrt5=[math]::Sqrt(5.0)
$nullCut=(($sqrt5-1.0)/$m)*[math]::Log($Nf*($sqrt5-1.0)/(2.0*$sqrt5)/$Kf)
$trueMean=$capture-2.0*(1.0-$capture)/$m
$trueVar=4.0*$capture*(1.0-$capture)/$m+12.0*[math]::Pow(1.0-$capture,2)/[math]::Pow($m,2)
$trueSd=[math]::Sqrt($trueVar)
$z=($trueMean-$nullCut)/$trueSd
$ret=$design.multiplicity_and_retention
Assert-Close ([double]$ret.analytic_null_cutoff) $nullCut 1e-15 'retention null cutoff'
Assert-Close ([double]$ret.analytic_true_mean_at_composite_capture) $trueMean 1e-15 'retention mean'
Assert-Close ([double]$ret.analytic_true_sd_at_composite_capture) $trueSd 1e-15 'retention sd'
Assert-Close ([double]$ret.analytic_z_margin) $z 1e-14 'retention z'
Assert-Eq ([bool]$ret.retention_is_distribution_free) $false 'retention not distribution-free'
Assert-Eq ([bool]$ret.exchangeable_randomization_p_value_claimed) $false 'no exchangeable p value'
Assert-Eq ([bool]$ret.familywise_p_value_claimed) $false 'no familywise p value'
$stress = $ret.finite_stress_design
Assert-Eq ([int]$stress.cell_count) 256 'finite retention cell count'
Assert-Eq ([int]$stress.planted_trials_per_cell) 1010 'retention trials per cell'
Assert-Close ([double]$stress.per_cell_bonferroni_alpha) (0.01/256.0) 1e-18 'Bonferroni alpha'
$lower=[math]::Pow(0.01/256.0,1.0/1010.0)
Assert-Close ([double]$stress.all_success_exact_lower_bound) $lower 1e-15 'all-success exact lower bound'
Assert-True ($lower -ge 0.99) 'simultaneous retention lower bound clears 0.99'
$coverage = @{
    abi=(HashSet-Ordinal); concentration=(HashSet-Ordinal); role=(HashSet-Ordinal); rho=(HashSet-Ordinal);
    shared=(HashSet-Ordinal); scale=(HashSet-Ordinal); residual=(HashSet-Ordinal)
}
for ($i=0; $i -lt 256; $i++) {
    [void]$coverage.abi.Add([string]($i%3))
    [void]$coverage.concentration.Add([string]((7*$i+[math]::Floor($i/3))%5))
    [void]$coverage.role.Add([string]((11*$i+[math]::Floor($i/5))%5))
    [void]$coverage.rho.Add([string]((13*$i+[math]::Floor($i/7))%5))
    [void]$coverage.shared.Add([string]((5*$i+[math]::Floor($i/11))%3))
    [void]$coverage.scale.Add([string]((7*$i+[math]::Floor($i/13))%3))
    [void]$coverage.residual.Add([string]((11*$i+[math]::Floor($i/17))%3))
}
Assert-Eq $coverage.abi.Count 3 'stress ABI value coverage'
Assert-Eq $coverage.concentration.Count 5 'stress concentration coverage'
Assert-Eq $coverage.role.Count 5 'stress role-ratio coverage'
Assert-Eq $coverage.rho.Count 5 'stress rho coverage'
Assert-Eq $coverage.shared.Count 3 'stress shared-correlation coverage'
Assert-Eq $coverage.scale.Count 3 'stress scale coverage'
Assert-Eq $coverage.residual.Count 3 'stress residual coverage'

# Threshold, rate/read and descriptor accounting.
$phys = $design.physical_ledger
Assert-Eq ([int64]$phys.target_weights) ([int64]18*1572864) 'target weights'
Assert-Eq ([int64]$phys.total_side_metadata_bytes) ([int64]8+18*4) 'total metadata bytes'
$sideBpw = 80.0*8.0/(18.0*1572864.0)
$composite = 1.0-(0.8/0.936397621)*[math]::Pow(2.0,-2.0*$sideBpw)
$standalone = 1.0-(0.8/0.9888693569009007)*[math]::Pow(2.0,-2.0*$sideBpw)
Assert-Close ([double]$phys.side_bpw) $sideBpw 1e-18 'side bpw'
Assert-Close ([double]$phys.metadata_adjusted_composite_required_capture) $composite 1e-15 'composite threshold'
Assert-Close ([double]$phys.metadata_adjusted_standalone_required_capture) $standalone 1e-15 'standalone threshold'
Assert-Close ([double]$phys.maximum_compatible_base_codec_bpw_at_2_15) (2.15-$sideBpw) 1e-15 'maximum base bpw'
$bytes215=4718592.0*2.15/8.0
$read215=1.169444+20.0/$bytes215
Assert-Close ([double]$phys.bytes_per_expert_at_2_15_bpw) $bytes215 1e-9 'expert bytes 2.15'
Assert-Close ([double]$phys.appended_cold_read_amplification_at_2_15_bpw) $read215 1e-15 'read amp 2.15'
Assert-True ($read215 -lt 2.0) 'read amplification arithmetic below 2x'
Assert-True ([double]$phys.metadata_adjusted_standalone_required_capture -gt [double]$phys.metadata_adjusted_composite_required_capture) 'standalone threshold strictly stronger'

$cal = $design.source_free_calibration
Assert-Close ([double]$cal.historical_raw_floor_seconds_for_stage0) ($stage0Values/46532862804.4677) 1e-9 'raw floor'
Assert-Close ([double]$cal.minimum_candidates_per_second_for_900_seconds) ($candidateCount/900.0) 1e-6 'candidate throughput gate'
Assert-Close ([double]$cal.minimum_anchor_values_per_second_for_900_seconds) ($stage0Values/900.0) 1e-3 'anchor throughput gate'
Assert-Close ([double]$cal.minimum_normal4_bundles_per_second_for_900_seconds) ($stage0Cores/900.0) 1e-3 'normal4 throughput gate'
Assert-Close ([double]$cal.minimum_domain_cross_fmas_per_second_for_900_seconds) (($candidateCount*33*1024)/900.0) 1e-2 'cross-FMA throughput gate'
Assert-Close ([double]$cal.maximum_average_wall_seconds_per_2pow24_shard_for_900_seconds) (900.0/768.0) 1e-15 'average shard wall gate'
Assert-True ([int64]$cal.peak_live_vram_limit_bytes -le [int64]8589934592) 'VRAM gate at most 8 GiB'
Assert-True ([int64]$cal.full_state_disk_limit_bytes -le [int64]4294967296) 'state disk gate at most 4 GiB'
$shape=$cal.non_authoritative_surrogate_full_shard_result
Assert-Eq $shape.status 'BLOCKED_SURROGATE_NOT_A_V1_RUNTIME_DECISION' 'surrogate runtime status'
Assert-Eq ([int64]$shape.shard_candidates) ([int64]16777216) 'exact-shape shard candidates'
Assert-Eq ([int]$shape.normal4_bundles_per_candidate) 256 'exact-shape bundles'
Assert-Eq ([int]$shape.anchor_values_per_candidate) 1024 'exact-shape values'
Assert-Eq ([int]$shape.domain_count) 33 'exact-shape domains'
Assert-Eq ([int]$shape.registers_per_thread) 106 'exact-shape registers'
Assert-Eq ([int64]$shape.local_spill_bytes) 0 'exact-shape local spills'
Assert-Eq ([int64]$shape.q_buffer_bytes) ([int64]33*16777216*4) 'exact-shape q bytes'
Assert-Close ([double]$shape.projected_768_shard_kernel_seconds) ([double]$shape.median_shard_kernel_seconds*768.0) 1e-6 'exact-shape kernel projection'
Assert-Close ([double]$shape.projected_768_shard_end_to_end_seconds) ([double]$shape.median_shard_end_to_end_seconds*768.0) 1e-6 'exact-shape e2e projection'
Assert-True ([double]$shape.projected_768_shard_end_to_end_seconds -gt [double]$cal.projected_post_compile_wall_limit_seconds) 'surrogate projection exceeds prospective gate'
Assert-Eq ([bool]$shape.model_or_qwen_access) $false 'surrogate no model access'
Assert-Eq @($shape.architecture_mismatches).Count 6 'surrogate mismatch disclosure count'
$verdict=$design.final_verdict
Assert-Eq $verdict.status 'EARLY_KILL_RUNTIME_NO_QWEN_PENDING_INDEPENDENT_DIRECT_CALIBRATION_AUDIT' 'final verdict status'
Assert-Close ([double]$verdict.direct_warm_over_limit_factor) ([double]$verdict.direct_warm_end_to_end_projection_seconds/900.0) 1e-12 'direct over-limit factor'
Assert-Eq ([bool]$verdict.qwen_or_model_access) $false 'verdict no model access'
Assert-Eq ([bool]$verdict.initializer_family_scientifically_tested) $false 'no initializer science claimed'
$direct=$cal.exact_direct_counter_full_shard_result
Assert-Eq $direct.status 'FAIL_RUNTIME_GATE_PENDING_INDEPENDENT_AUDIT' 'direct runtime status'
Assert-Eq ([int64]$direct.shard_candidates) ([int64]16777216) 'direct shard candidates'
Assert-Eq ([int]$direct.normal4_bundles_per_candidate) 256 'direct bundles'
Assert-Eq ([int]$direct.anchor_values_per_candidate) 1024 'direct values'
Assert-Eq ([int]$direct.domain_count) 33 'direct domains'
Assert-Eq ([int]$direct.repetitions) 3 'direct repetitions'
Assert-Eq ([int]$direct.registers_per_thread) 108 'direct registers'
Assert-Eq ([int64]$direct.local_spill_bytes) 0 'direct spill bytes'
Assert-Eq ([int64]$direct.q_buffer_bytes) ([int64]33*16777216*4) 'direct q bytes'
Assert-Close ([double]$direct.projected_768_shard_kernel_seconds) ([double]$direct.median_shard_kernel_seconds*768.0) 1e-9 'direct kernel projection'
Assert-Close ([double]$direct.projected_768_shard_warm_end_to_end_seconds) (([double]$direct.median_shard_kernel_seconds+[double]$direct.median_warm_selection_seconds)*768.0+[double]$direct.one_time_cold_selection_excess_seconds) 1e-9 'direct warm projection'
Assert-True ([double]$direct.projected_768_shard_warm_end_to_end_seconds -gt [double]$cal.projected_post_compile_wall_limit_seconds) 'direct runtime gate exceeded'
Assert-Eq ([int]$direct.deterministic_q_sentinel_replays) 3 'direct q replay count'
Assert-Eq ([int]$direct.deterministic_topk_hash_replays) 3 'direct TopK replay count'
Assert-Eq ([int]$direct.performance_kernel_curand_init_calls) 0 'direct kernel curand_init absent'
Assert-Eq ([int]$direct.performance_kernel_curand_normal4_calls) 0 'direct kernel curand_normal4 absent'
Assert-Eq ([int]$direct.performance_kernel_direct_philox_calls_per_bundle) 1 'direct Philox calls'
Assert-Eq ([int]$direct.performance_kernel_box_muller_pair_calls_per_bundle) 2 'direct Box-Muller calls'
Assert-Eq ([int]$direct.parity_rows) 132 'direct parity rows'
Assert-Eq ([bool]$direct.raw_bitwise_equal) $true 'direct raw parity'
Assert-Eq ([bool]$direct.scaled_bf16_bitwise_equal) $true 'direct BF16 parity'
Assert-Eq ([bool]$direct.terminal_counter_equal) $true 'direct counter parity'
Assert-Eq ([bool]$direct.model_or_qwen_access) $false 'direct no model access'
Assert-Eq ([bool]$direct.omitted_work_can_only_increase_projection) $true 'omitted work direction'

$prior = $design.prior_family_relation
Assert-Eq ([int64]$prior.procedural_anchor_expansion_effective_candidates) 3266322626690 'procedural comparator count'
Assert-Eq ([int64]$prior.fuseed_candidate_count) $candidateCount 'FUSEED comparator count'
Assert-Close ([double]$prior.candidate_count_reduction_factor) (3266322626690.0/$candidateCount) 1e-9 'candidate reduction factor'
Assert-Eq ([int64]$prior.direct_rng_state_envelope_candidates) 125862912 'direct comparator count'

# Every dependency is an explicit hash-bound research artifact; no discovery outside research.
$researchRoot = [System.IO.Path]::GetFullPath((Join-Path $root '..'))
foreach ($dep in @($sources.dependencies)) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $root ([string]$dep.path)))
    Assert-True ($candidate.StartsWith($researchRoot+[System.IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) "dependency stays in research: $($dep.id)"
    Assert-True (Test-Path -LiteralPath $candidate -PathType Leaf) "dependency exists: $($dep.id)"
    Assert-Eq (File-Sha256 $candidate) ([string]$dep.sha256) "dependency hash: $($dep.id)"
}

# Independently bind the decisive direct-counter result fields rather than
# trusting only values copied into this lock.
$directDep = @($sources.dependencies | Where-Object id -eq 'exact_direct_counter_result')
Assert-Eq $directDep.Count 1 'one direct result dependency'
$directResultPath = [System.IO.Path]::GetFullPath((Join-Path $root ([string]$directDep[0].path)))
$directResult = Get-Content -LiteralPath $directResultPath -Raw | ConvertFrom-Json -Depth 100
Assert-Eq $directResult.schema 'fuseed_u32_source_free_direct_counter_calibration_v0' 'direct result schema'
Assert-Eq ([int]$directResult.shape.candidates) 16777216 'direct result candidate shape'
Assert-Eq ([int]$directResult.shape.normal4_bundles_per_candidate) 256 'direct result bundle shape'
Assert-Eq ([int]$directResult.shape.normal_values_per_candidate) 1024 'direct result value shape'
Assert-Eq ([int]$directResult.shape.domains) 33 'direct result domain shape'
Assert-Eq ([int]$directResult.shape.repetitions) 3 'direct result repetitions'
Assert-Eq ([int64]$directResult.shape.q_bytes) ([int64]33*16777216*4) 'direct result q bytes'
Assert-Eq ([int]$directResult.direct_source.curand_init_calls_in_performance_kernel) 0 'result no curand_init'
Assert-Eq ([int]$directResult.direct_source.curand_normal4_calls_in_performance_kernel) 0 'result no curand_normal4'
Assert-Eq ([int]$directResult.direct_source.direct_philox_calls_in_performance_kernel) 1 'result one direct Philox source occurrence'
Assert-Eq ([int]$directResult.direct_source.box_muller_pair_calls_in_performance_kernel) 2 'result two Box-Muller source occurrences'
Assert-Eq ([string]$directResult.direct_source.derived_cuda_sha256) ([string]$direct.derived_cuda_sha256) 'direct CUDA hash'
Assert-Eq ([int]$directResult.parity.rows) 132 'result parity rows'
foreach ($field in @('vector_sha256','raw_float32_sha256','scaled_widened_bf16_sha256','direct_counter_sha256','terminal_counter_sha256')) {
    $lockedName = if ($field -eq 'vector_sha256') {'parity_vector_sha256'} else {$field}
    Assert-Eq ([string]$directResult.parity.$field) ([string]$direct.$lockedName) "direct parity hash $field"
}
Assert-Eq ([bool]$directResult.parity.raw_bitwise_equal) $true 'result raw bitwise parity'
Assert-Eq ([bool]$directResult.parity.scaled_bf16_bitwise_equal) $true 'result BF16 bitwise parity'
Assert-Eq ([bool]$directResult.parity.terminal_counter_equal) $true 'result terminal counter parity'
Assert-Close ([double]$directResult.aggregate.median_kernel_seconds_per_shard) ([double]$direct.median_shard_kernel_seconds) 1e-15 'result median kernel'
Assert-Close ([double]$directResult.aggregate.median_warm_selection_seconds_per_shard) ([double]$direct.median_warm_selection_seconds) 1e-15 'result warm selection'
Assert-Close ([double]$directResult.aggregate.one_time_cold_selection_excess_seconds) ([double]$direct.one_time_cold_selection_excess_seconds) 1e-15 'result cold excess'
Assert-Close ([double]$directResult.aggregate.projected_three_abi_kernel_seconds) ([double]$direct.projected_768_shard_kernel_seconds) 1e-12 'result kernel projection'
Assert-Close ([double]$directResult.aggregate.projected_three_abi_warm_end_to_end_seconds_excluding_journal) ([double]$direct.projected_768_shard_warm_end_to_end_seconds) 1e-12 'result warm projection'
Assert-Eq ([bool]$directResult.aggregate.replay_deterministic) $true 'result deterministic'
Assert-Eq ([bool]$directResult.aggregate.kernel_projection_below_gate) $false 'result kernel fails gate'
Assert-Eq ([bool]$directResult.aggregate.warm_e2e_projection_below_gate) $false 'result warm e2e fails gate'
Assert-Eq ([int]$directResult.kernel.attributes.num_regs) 108 'result register count'
Assert-Eq ([int64]$directResult.kernel.attributes.local_size_bytes) 0 'result local size'
Assert-Eq ([int]@($directResult.cuda_headers.PSObject.Properties).Count) 5 'five CUDA headers bound'
Assert-Eq ([int]@($directResult.rows).Count) 3 'three timing rows'
$seedHashes=@($directResult.rows | ForEach-Object domain_topk_seed_sha256 | Select-Object -Unique)
$valueHashes=@($directResult.rows | ForEach-Object domain_topk_value_sha256 | Select-Object -Unique)
$qHashes=@($directResult.rows | ForEach-Object q_sentinel_sha256 | Select-Object -Unique)
Assert-Eq $seedHashes.Count 1 'three direct seed hashes identical'
Assert-Eq $valueHashes.Count 1 'three direct value hashes identical'
Assert-Eq $qHashes.Count 1 'three direct q hashes identical'

foreach ($attestation in @($design.access_attestation,$sources.access_attestation)) {
    foreach ($prop in $attestation.PSObject.Properties) {
        Assert-Eq ([bool]$prop.Value) $false "access attestation $($prop.Name)"
    }
}
Assert-Eq ([bool]$design.external_evidence_attestation.bound_synthetic_calibration_used_cupy_cuda_gpu) $true 'external calibration used GPU disclosed'
Assert-Eq ([bool]$design.external_evidence_attestation.bound_synthetic_calibration_accessed_model_or_qwen) $false 'external calibration no model access'
Assert-Eq ([bool]$design.external_evidence_attestation.bound_synthetic_calibration_authorizes_v1_decision) $false 'external calibration no v1 authority'
Assert-Eq ([bool]$design.external_evidence_attestation.bound_direct_calibration_used_cupy_cuda_gpu) $true 'direct calibration GPU disclosed'
Assert-Eq ([bool]$design.external_evidence_attestation.bound_direct_calibration_accessed_model_or_qwen) $false 'direct calibration no model access'
Assert-Eq ([bool]$design.external_evidence_attestation.bound_direct_calibration_decision_pending_independent_audit) $true 'direct decision audit pending'

$receiptPath = Join-Path $root 'audit_receipt.json'
if (-not $SkipReceiptAndManifest) {
    Assert-True (Test-Path -LiteralPath $receiptPath -PathType Leaf) 'self-verification receipt exists'
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json -Depth 50
    Assert-Eq $receipt.schema 'fuseed_u32_self_verification_receipt_v1' 'receipt schema'
    Assert-Eq $receipt.status 'PASS_SOURCE_ONLY_EARLY_KILL_PENDING_INDEPENDENT_AUDIT_NOT_EXECUTION_AUTHORITY' 'receipt status'
    Assert-Eq ([string]$receipt.verification.plan_sha256) $planDigest 'receipt plan digest'
    Assert-Eq ([string]$receipt.verification.stage0_plan_sha256) $stage0Digest 'receipt stage0 digest'
    Assert-Eq ([int64]$receipt.verification.candidate_count) $candidateCount 'receipt candidate count'
    Assert-Eq ([int64]$receipt.verification.bundle_attempts) $attempts 'receipt bundle attempts'
    foreach ($name in @('README.md','design_lock.json','source_bindings.json','verify_design.ps1')) {
        Assert-Eq (File-Sha256 (Join-Path $root $name)) ([string]$receipt.file_sha256.$name) "receipt file hash $name"
    }
    $payloadLines = @(
        "schema=$($receipt.schema)",
        "status=$($receipt.status)",
        "checks=$([int64]$receipt.verification.preseal_checks)",
        "candidate_count=$([int64]$receipt.verification.candidate_count)",
        "plan_sha256=$($receipt.verification.plan_sha256)",
        "stage0_plan_sha256=$($receipt.verification.stage0_plan_sha256)",
        "bundle_attempts=$([int64]$receipt.verification.bundle_attempts)",
        "README.md=$($receipt.file_sha256.'README.md')",
        "design_lock.json=$($receipt.file_sha256.'design_lock.json')",
        "source_bindings.json=$($receipt.file_sha256.'source_bindings.json')",
        "verify_design.ps1=$($receipt.file_sha256.'verify_design.ps1')",
        "local_design_access_all_false=$(([bool]$receipt.verification.local_design_access_all_false).ToString().ToLowerInvariant())",
        "external_bound_calibration_gpu_access=$(([bool]$receipt.verification.external_bound_calibration_gpu_access).ToString().ToLowerInvariant())"
    )
    Assert-Eq (Text-Sha256 (($payloadLines -join "`n")+"`n")) ([string]$receipt.receipt_payload_sha256) 'receipt internal hash'

    $manifestPath = Join-Path $root 'ARTIFACT_SHA256SUMS.txt'
    Assert-True (Test-Path -LiteralPath $manifestPath -PathType Leaf) 'manifest exists'
    $expectedMembers = @('ARTIFACT_SHA256SUMS.txt','README.md','audit_receipt.json','design_lock.json','source_bindings.json','verify_design.ps1')
    $actualMembers = @(Get-ChildItem -LiteralPath $root -Force | ForEach-Object Name | Sort-Object)
    Assert-Eq $actualMembers.Count $expectedMembers.Count 'package member count'
    foreach ($name in $expectedMembers) { Assert-True ($actualMembers -contains $name) "package member $name" }
    $manifestRows = @{}
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        Assert-True ($line -match '^([0-9a-f]{64})  (README\.md|audit_receipt\.json|design_lock\.json|source_bindings\.json|verify_design\.ps1)$') "manifest syntax $line"
        $hash=$Matches[1];$name=$Matches[2]
        Assert-True (-not $manifestRows.ContainsKey($name)) "manifest no duplicate $name"
        $manifestRows[$name]=$hash
    }
    Assert-Eq $manifestRows.Count 5 'manifest entry count'
    foreach ($name in @('README.md','audit_receipt.json','design_lock.json','source_bindings.json','verify_design.ps1')) {
        Assert-Eq $manifestRows[$name] (File-Sha256 (Join-Path $root $name)) "manifest hash $name"
    }
}

[pscustomobject]@{
    status='PASS'
    checks=$script:Checks
    plan_sha256=$planDigest
    stage0_plan_sha256=$stage0Digest
    bundle_attempts=$attempts
    candidate_count=$candidateCount
    model_access=$false
    design_author_gpu_access=$false
    external_bound_calibration_gpu_access=$true
    network_access=$false
} | ConvertTo-Json -Depth 5
