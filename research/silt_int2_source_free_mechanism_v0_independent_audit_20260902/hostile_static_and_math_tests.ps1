$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Source-free hostile audit.  This script reads only the unsealed SILT source
# directory and the public universal contract.  It never discovers or opens a
# model, current-container, or matched-control payload.

$script:Checks = [System.Collections.Generic.List[object]]::new()

function Check([bool]$Condition, [string]$Name, [object]$Evidence = $null) {
    if (-not $Condition) {
        throw "FAILED: $Name :: $Evidence"
    }
    $script:Checks.Add([ordered]@{ name = $Name; status = "PASS"; evidence = $Evidence })
}

function Mod([int]$Value, [int]$Alphabet) {
    return (($Value % $Alphabet) + $Alphabet) % $Alphabet
}

function Forward-Pair([int]$Left, [int]$Right, [int]$Alphabet, [int]$Code) {
    $swap = ($Code -shr 2) -band 1
    $p = ($Code -shr 1) -band 1
    $u = $Code -band 1
    if ($swap -eq 1) { $x = $Right; $y = $Left } else { $x = $Left; $y = $Right }
    $d = Mod ($y - $p * $x) $Alphabet
    $c = Mod ($x + $u * $d) $Alphabet
    return [int[]]@($c, $d)
}

function Inverse-Pair([int]$Coarse, [int]$Detail, [int]$Alphabet, [int]$Code) {
    $swap = ($Code -shr 2) -band 1
    $p = ($Code -shr 1) -band 1
    $u = $Code -band 1
    $x = Mod ($Coarse - $u * $Detail) $Alphabet
    $y = Mod ($Detail + $p * $x) $Alphabet
    if ($swap -eq 1) { return [int[]]@($y, $x) }
    return [int[]]@($x, $y)
}

function Tree-Forward([int[]]$Values, [int]$Alphabet, [int[]]$Permutation, [int[]]$Selectors) {
    $current = [System.Collections.Generic.List[int]]::new()
    foreach ($index in $Permutation) { $current.Add($Values[$index]) }
    $levels = [System.Collections.Generic.List[object]]::new()
    $offset = 0
    while ($current.Count -gt 1) {
        $pairs = [int][Math]::Floor($current.Count / 2)
        $details = [int[]]::new($pairs)
        $next = [System.Collections.Generic.List[int]]::new()
        for ($pair = 0; $pair -lt $pairs; $pair++) {
            $mapped = Forward-Pair $current[2 * $pair] $current[2 * $pair + 1] $Alphabet $Selectors[$offset + $pair]
            $next.Add($mapped[0])
            $details[$pair] = $mapped[1]
        }
        if (($current.Count -band 1) -eq 1) { $next.Add($current[$current.Count - 1]) }
        $levels.Add($details)
        $current = $next
        $offset += $pairs
    }
    Check ($offset -eq $Values.Count - 1) "tree emits L-1 details" @{ lanes = $Values.Count; details = $offset }
    return [pscustomobject]@{ Root = $current[0]; Levels = $levels.ToArray() }
}

function Tree-Inverse([int]$Root, [object[]]$Levels, [int]$Lanes, [int]$Alphabet, [int[]]$Permutation, [int[]]$Selectors) {
    $widths = [System.Collections.Generic.List[int]]::new()
    $pairsAtDepth = [System.Collections.Generic.List[int]]::new()
    $width = $Lanes
    while ($width -gt 1) {
        $widths.Add($width)
        $pairsAtDepth.Add([int][Math]::Floor($width / 2))
        $width = [int][Math]::Floor($width / 2) + ($width -band 1)
    }
    $offsets = [int[]]::new($pairsAtDepth.Count + 1)
    for ($i = 0; $i -lt $pairsAtDepth.Count; $i++) { $offsets[$i + 1] = $offsets[$i] + $pairsAtDepth[$i] }
    $current = [System.Collections.Generic.List[int]]::new()
    $current.Add($Root)
    for ($depth = $pairsAtDepth.Count - 1; $depth -ge 0; $depth--) {
        $pairs = $pairsAtDepth[$depth]
        $previous = [int[]]::new($widths[$depth])
        $details = [int[]]$Levels[$depth]
        for ($pair = 0; $pair -lt $pairs; $pair++) {
            $mapped = Inverse-Pair $current[$pair] $details[$pair] $Alphabet $Selectors[$offsets[$depth] + $pair]
            $previous[2 * $pair] = $mapped[0]
            $previous[2 * $pair + 1] = $mapped[1]
        }
        if (($widths[$depth] -band 1) -eq 1) { $previous[$previous.Count - 1] = $current[$pairs] }
        $current = [System.Collections.Generic.List[int]]::new()
        foreach ($value in $previous) { $current.Add($value) }
    }
    $output = [int[]]::new($Lanes)
    for ($i = 0; $i -lt $Lanes; $i++) { $output[$Permutation[$i]] = $current[$i] }
    return $output
}

function Factorial([int]$N) {
    [System.Numerics.BigInteger]$result = 1
    for ($i = 2; $i -le $N; $i++) { $result *= $i }
    return $result
}

function Permutation-Byte-Count([int]$N) {
    [System.Numerics.BigInteger]$value = (Factorial $N) - 1
    $bits = 0
    while ($value -gt 0) { $bits++; $value = $value -shr 1 }
    return [int][Math]::Floor(($bits + 7) / 8)
}

function Rank-Permutation([int[]]$Permutation) {
    $available = [System.Collections.ArrayList]::new()
    for ($i = 0; $i -lt $Permutation.Count; $i++) { [void]$available.Add($i) }
    [System.Numerics.BigInteger]$rank = 0
    for ($position = 0; $position -lt $Permutation.Count; $position++) {
        $index = $available.IndexOf($Permutation[$position])
        if ($index -lt 0) { throw "duplicate permutation" }
        $rank += [System.Numerics.BigInteger]$index * (Factorial ($Permutation.Count - $position - 1))
        $available.RemoveAt($index)
    }
    return $rank
}

function Unrank-Permutation([int]$N, [System.Numerics.BigInteger]$Rank) {
    $available = [System.Collections.ArrayList]::new()
    for ($i = 0; $i -lt $N; $i++) { [void]$available.Add($i) }
    $result = [int[]]::new($N)
    for ($position = 0; $position -lt $N; $position++) {
        $factorial = Factorial ($N - $position - 1)
        $index = [int]($Rank / $factorial)
        $Rank = $Rank % $factorial
        $result[$position] = [int]$available[$index]
        $available.RemoveAt($index)
    }
    return $result
}

function Q16-Row([long[]]$Counts) {
    $total = 65536L
    $remaining = $total - $Counts.Count
    $denominator = $Counts.Count
    foreach ($count in $Counts) { $denominator += $count }
    $frequencies = [long[]]::new($Counts.Count)
    $remainders = [long[]]::new($Counts.Count)
    for ($i = 0; $i -lt $Counts.Count; $i++) {
        $adjusted = $Counts[$i] + 1
        $frequencies[$i] = [Math]::Floor($adjusted * $remaining / $denominator) + 1
        $remainders[$i] = ($adjusted * $remaining) % $denominator
    }
    $missing = $total - ($frequencies | Measure-Object -Sum).Sum
    $order = 0..($Counts.Count - 1) | Sort-Object @{Expression={-$remainders[$_]}}, @{Expression={$_}}
    for ($i = 0; $i -lt $missing; $i++) { $frequencies[$order[$i]]++ }
    return $frequencies
}

$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$producer = Join-Path $repo "research\silt_int2_source_free_mechanism_v0"
$contractPath = Join-Path $repo "docs\UNIVERSAL_SWIGLU_MOE_CODEC_CONTRACT.md"
Check (Test-Path -LiteralPath $producer -PathType Container) "producer directory exists" $producer
Check (Test-Path -LiteralPath $contractPath -PathType Leaf) "universal contract exists" $contractPath

$expectedHashes = [ordered]@{
    "cupy_backend.py" = "bc86e60092913ec0d6fda3c740c773f94369b9643fd162c97797b802c2ed6ada"
    "design_lock.json" = "255955d838892cb98761660700eec504dbe521e474351617c8077424800c9ef2"
    "independent_decoder.py" = "854e49590f4917aed703ccd0aa310bcbcc60ec04f454109440b8db1c3d9454ce"
    "README.md" = "00cc671ae94f86d37297c444369093299f30fbd288909f58f7b6da61786c5ac3"
    "run_synthetic.py" = "c831bcb1878923508c490342ad79146fc905643c81d513c97a72df115262364e"
    "silt_mechanism.py" = "8965bec994c37d37bf720d75593224be6c86e499def34c9191068375f829cf7d"
    "test_source_only.py" = "b7007eb3a61d1933a87868da6bcbbdbc3d0928df71f468b360f3abdb9bdea4cb"
    "verify_source.py" = "593ebfe7e4fe0fce9fa3053aabc5d560827397daf5147c17ae6e3d6c4f8cbe09"
}
$actualNames = @(Get-ChildItem -LiteralPath $producer -File | Sort-Object Name | ForEach-Object Name)
Check (($actualNames -join "|") -eq (($expectedHashes.Keys | Sort-Object) -join "|")) "all and only observed producer files covered" $actualNames
foreach ($name in $expectedHashes.Keys) {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $producer $name)).Hash.ToLowerInvariant()
    Check ($hash -eq $expectedHashes[$name]) "observed source hash unchanged: $name" $hash
}

$design = Get-Content -Raw -LiteralPath (Join-Path $producer "design_lock.json") | ConvertFrom-Json
Check ($design.status -eq "SOURCE_ONLY_UNSEALED_NO_PAYLOAD_AUTHORITY") "design is explicitly unsealed/no authority" $design.status
Check (-not [bool]$design.scope.payload_authority -and -not [bool]$design.scope.result_authority) "design disclaims payload/result authority"

# Exhaust every scalar pair and selector.  Pair bijectivity is sufficient for
# the recursively composed tree; the following arbitrary-width tests exercise
# ordering, odd carries, and factoradic-style leaf permutations independently.
$signatures = @{}
foreach ($alphabet in @(2, 4)) {
    $seen = @{}
    for ($code = 0; $code -lt 8; $code++) {
        $signatureParts = [System.Collections.Generic.List[string]]::new()
        for ($left = 0; $left -lt $alphabet; $left++) {
            for ($right = 0; $right -lt $alphabet; $right++) {
                $forward = Forward-Pair $left $right $alphabet $code
                $inverse = Inverse-Pair $forward[0] $forward[1] $alphabet $code
                Check ($inverse[0] -eq $left -and $inverse[1] -eq $right) "pair inverse A=$alphabet code=$code x=$left y=$right"
                $signatureParts.Add("$($forward[0]),$($forward[1])")
            }
        }
        $signature = $signatureParts -join ";"
        if (-not $seen.ContainsKey($signature)) { $seen[$signature] = [System.Collections.Generic.List[int]]::new() }
        $seen[$signature].Add($code)
    }
    $aliases = @($seen.Values | Where-Object Count -gt 1 | ForEach-Object { @($_) -join "," } | Sort-Object)
    if ($alphabet -eq 2) {
        Check ($seen.Count -eq 6) "GF2 has only six unique selector transforms" @{ unique = $seen.Count; aliases = $aliases }
        Check (($aliases -join "|") -eq "2,7|3,6") "exact GF2 selector aliases" $aliases
    } else {
        Check ($seen.Count -eq 8 -and $aliases.Count -eq 0) "Z4 keeps all eight selector transforms distinct" @{ unique = $seen.Count; aliases = $aliases }
    }
}

foreach ($alphabet in @(2, 4)) {
    foreach ($lanes in @(1, 2, 3, 5, 17, 97, 257)) {
        $values = [int[]]::new($lanes)
        for ($i = 0; $i -lt $lanes; $i++) { $values[$i] = ($i * 17 + $lanes) % $alphabet }
        foreach ($reverse in @($false, $true)) {
            $permutation = [int[]](0..($lanes - 1))
            if ($reverse) { [Array]::Reverse($permutation) }
            foreach ($code in 0..7) {
                $selectors = if ($lanes -gt 1) { [int[]](0..($lanes - 2) | ForEach-Object { $code }) } else { [int[]]@() }
                $lifted = Tree-Forward $values $alphabet $permutation $selectors
                $rebuilt = Tree-Inverse $lifted.Root $lifted.Levels $lanes $alphabet $permutation $selectors
                Check (($values -join ",") -eq ($rebuilt -join ",")) "tree roundtrip A=$alphabet L=$lanes reverse=$reverse code=$code"
            }
        }
    }
}

$random = [Random]::new(9022026)
foreach ($n in 1..12) {
    for ($trial = 0; $trial -lt 25; $trial++) {
        $permutation = [int[]](0..($n - 1))
        for ($i = $n - 1; $i -gt 0; $i--) {
            $j = $random.Next($i + 1); $tmp = $permutation[$i]; $permutation[$i] = $permutation[$j]; $permutation[$j] = $tmp
        }
        $rank = Rank-Permutation $permutation
        $rebuilt = Unrank-Permutation $n $rank
        Check (($permutation -join ",") -eq ($rebuilt -join ",")) "factoradic roundtrip n=$n trial=$trial" $rank.ToString()
        Check ($rank -ge 0 -and $rank -lt (Factorial $n)) "factoradic rank bound n=$n trial=$trial"
    }
}
Check ((Permutation-Byte-Count 1) -eq 0) "factoradic byte width n=1" 0
Check ((Permutation-Byte-Count 5) -eq 1 -and (Permutation-Byte-Count 6) -eq 2) "factoradic whole-byte boundary n=5/6"

foreach ($alphabet in @(2, 4)) {
    for ($trial = 0; $trial -lt 100; $trial++) {
        $counts = [long[]]::new($alphabet)
        for ($i = 0; $i -lt $alphabet; $i++) { $counts[$i] = $random.Next(0, 1000000) }
        $row = Q16-Row $counts
        Check ((($row | Measure-Object -Sum).Sum) -eq 65536) "Q16 exact sum A=$alphabet trial=$trial"
        Check (($row | Measure-Object -Minimum).Minimum -ge 1 -and ($row | Measure-Object -Maximum).Maximum -le 65535) "Q16 uint16 positive range A=$alphabet trial=$trial"
    }
}

$sm = Get-Content -Raw -LiteralPath (Join-Path $producer "silt_mechanism.py")
$ind = Get-Content -Raw -LiteralPath (Join-Path $producer "independent_decoder.py")
$gpu = Get-Content -Raw -LiteralPath (Join-Path $producer "cupy_backend.py")
$runner = Get-Content -Raw -LiteralPath (Join-Path $producer "run_synthetic.py")
$verifier = Get-Content -Raw -LiteralPath (Join-Path $producer "verify_source.py")
$readme = Get-Content -Raw -LiteralPath (Join-Path $producer "README.md")
$contract = Get-Content -Raw -LiteralPath $contractPath

# The literal struct is 104 bytes, leaving only 249 16-byte directory rows.
$globalStructBytes = 104
$directoryEntryBytes = 16
$directoryCapacity = [Math]::Floor((4096 - $globalStructBytes) / $directoryEntryBytes)
Check ($directoryCapacity -eq 249) "global header directory capacity counterexample" @{ capacity = $directoryCapacity; first_unsupported_positive_experts = 250 }
Check ($design.universality.allowed_decoder_inputs -contains "positive lane/vector/expert counts") "design advertises unrestricted positive expert count"
Check ($sm -match 'require\(len\(packed\) \+ len\(entries\) <= GLOBAL_HEADER_BYTES, "global directory fit"\)') "producer rejects only after building over-capacity directory"

# Concrete anti-gaming/ownership counterexample using legal 4-KiB page sizes.
$E = 8.0; $G = 8192.0; $frames = @(4096.0) + @(1..7 | ForEach-Object { 8192.0 })
$T = $G + (($frames | Measure-Object -Sum).Sum)
$fair = $T / $E
$reportedMax = ($frames | ForEach-Object { ($G + $_) / $fair } | Measure-Object -Maximum).Maximum
$ownerSmall = ($G + $frames[0]) / ($G / $E + $frames[0])
Check ($reportedMax -lt 2.0 -and [Math]::Abs($reportedMax - (32.0/17.0)) -lt 1e-12) "current total/E ledger passes unequal frames" @{ reported_max = $reportedMax }
Check ([Math]::Abs($ownerSmall - 2.4) -lt 1e-12 -and $ownerSmall -gt 2.0) "owner-aware ledger fails same expert" @{ owner_aware_expert0 = $ownerSmall; G = $G; frames = $frames }
Check ($sm -match 'fair_share = len\(packet\) / expert_count' -and $sm -match '"amplification": cold / fair_share') "producer contains vulnerable total/E denominator"

# Parser/canonicality/resource-order findings are tested as exact source facts.
Check ($sm -match 'states = math\.factorial\(lanes\)' -and $sm -notmatch 'MAX_LANES') "producer factorial has no lane cap"
Check ($ind -match 'math\.factorial\(lanes\)' -and $ind -notmatch 'MAX_LANES') "independent decoder factorial has no lane cap"
Check ($sm -match 'roots = np\.empty\(root_count' -and $sm -match 'details = np\.empty\(detail_count') "producer allocates decoded counts without explicit byte cap"
Check ($ind -match 'roots = np\.empty\(roots_count' -and $ind -match 'details = np\.empty\(details_count') "independent decoder allocates decoded counts without explicit byte cap"
Check ($ind -match 'for index in range\(experts\)' -and $ind -notmatch 'directory_end <= GLOBAL_HEADER') "independent directory unpack lacks prior header bound"
Check ($sm -match 'ArithmeticDecoder\(payload\)' -and $sm -notmatch 'ArithmeticDecoder\(payload, meaningful') "meaningful bit length is not passed to producer arithmetic decoder"
Check ($ind -match 'Decoder\(packet\)' -and $ind -notmatch 'Decoder\(packet, meaningful') "meaningful bit length is not passed to independent arithmetic decoder"
Check ($sm -match 'if self\.bit_index >= 8 \* len\(self\.payload\):[\s\S]*?return 0') "producer arithmetic decoder zero-extends beyond finite bytes"
Check ($ind -match 'if self\.position >= 8 \* len\(self\.packet\):[\s\S]*?return 0') "independent arithmetic decoder zero-extends beyond finite bytes"
Check ($sm -match 'meaningful_bits <= 8 \* payload_bytes' -and $sm -notmatch 'meaningful_bits > 0') "frame grammar accepts meaningful_bits=0"

# Root-of-trust, import, output, and telemetry checks.
Check ($verifier -match 'loadTestsFromName\("test_source_only"\)') "verifier imports test module from ambient sys.path"
Check ($runner -match '(?m)^from cupy_backend import' -and $runner -match '(?m)^from independent_decoder import' -and $runner -match '(?m)^from silt_mechanism import') "runner imports unauthenticated siblings" 
Check ($runner -match 'require\(not path\.exists\(\)' -and $runner -match 'path\.write_(text|bytes)') "output writes have exists/open TOCTOU"
Check ($runner -match 'output_dir\.resolve\(\)' -and $runner -notmatch 'O_NOFOLLOW|lstat|is_symlink') "output path has no symlink/held-directory defense"
Check ($gpu -match 'cp\.asarray\(np\.ascontiguousarray\(train_leaves\)\)' -and $gpu -match 'lift_forward_device') "CuPy path performs real source H2D and GPU lifting"
Check ($gpu -notmatch 'h2d_bytes') "GPU receipt omits exact H2D bytes"
Check ($gpu -notmatch 'peak_host_bytes|peak_rss_bytes|host_rss') "GPU receipt omits peak host memory"
Check ($gpu -match 'nvmlDeviceGetHandleByIndex\(self\.device_index\)') "NVML uses CUDA logical index directly"
Check ($gpu -match 'sampled_peak_device_used_bytes' -and $gpu -notmatch 'incremental_peak') "peak VRAM is total device-used, not run-attributable incremental peak"

$tracked = & git -C $repo ls-files --error-unmatch -- "research/silt_int2_source_free_mechanism_v0" 2>$null
Check ($LASTEXITCODE -ne 0 -or -not $tracked) "producer has no committed/public git root in this workspace" "untracked"
$python = Get-Command python, python3, py -ErrorAction SilentlyContinue
Check ($null -eq $python) "local runtime cannot execute producer Python/CuPy suite" "no python/python3/py on PATH"

Check ($readme -match 'not a source-gain or model-quality result' -and $readme -match 'does not imply that the dependency exists in SwiGLU-MoE weights') "README correctly limits synthetic result to mechanism proof"
Check ($contract -match 'Gate: \[intermediate, hidden\]' -and $contract -match 'Down: \[hidden, intermediate\]') "universal contract requires actual SwiGLU triplet portability"
Check ($sm -notmatch '(?i)swiglu|gate_matrix|down_matrix|up_matrix') "mechanism has no actual SwiGLU triplet adapter"

$receipt = [ordered]@{
    schema = "silt-v0-independent-hostile-static-math-audit-v0"
    status = "PASS_TEST_HARNESS_BLOCK_PRODUCER_PROMOTION"
    source_scope = "producer source plus universal contract only; no model/current/control payload"
    checks = $script:Checks.Count
    blockers_detected = @(
        "wrong unequal-expert cold denominator",
        "250-expert directory failure and unbounded geometry/resource order",
        "noncanonical arithmetic accepted by decode API",
        "unsealed ambient-import root of trust",
        "mandatory CuPy replay lacks exact public source provenance and is unexecuted",
        "telemetry lacks H2D bytes and host peak"
    )
    selector_unique_transforms = [ordered]@{ GF2 = 6; Z4 = 8 }
    selector_serialized_ids = 8
    directory_capacity = $directoryCapacity
    cold_counterexample = [ordered]@{ experts = 8; G = 8192; frames = $frames; reported_max = $reportedMax; owner_aware_expert0 = $ownerSmall }
}
$receipt | ConvertTo-Json -Depth 8
