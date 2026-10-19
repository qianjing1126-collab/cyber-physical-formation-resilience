param(
    [string]$DevelopmentRoot = "runs\g3\ns3_raw\G3B_ns3_confirmation_v1",
    [string]$HoldoutRoot = "runs\g3\ns3_raw\G3B2_ns3_holdout_v1",
    [string]$OutputRoot = "runs\g3\ns3_raw"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$development = Join-Path $projectRoot $DevelopmentRoot
$holdout = Join-Path $projectRoot $HoldoutRoot
$root = Join-Path $projectRoot $OutputRoot
$baseName = "G3B2_two_state_validation_v1"
$destination = Join-Path $root $baseName
$retry = 0
while (Test-Path -LiteralPath $destination) {
    $retry += 1
    $destination = Join-Path $root ("{0}_retry{1:D2}" -f $baseName, $retry)
}
New-Item -ItemType Directory -Path $destination | Out-Null

$regimes = @("light", "moderate", "heavy")
$developmentSeeds = 1..5
$holdoutSeeds = 16..25
foreach ($regime in $regimes) {
    foreach ($seed in $developmentSeeds) {
        $name = "{0}_seed{1:D4}" -f $regime, $seed
        Copy-Item -LiteralPath (Join-Path $development $name) -Destination (Join-Path $destination $name) -Recurse
    }
    foreach ($seed in $holdoutSeeds) {
        $name = "{0}_seed{1:D4}" -f $regime, $seed
        Copy-Item -LiteralPath (Join-Path $holdout $name) -Destination (Join-Path $destination $name) -Recurse
    }
}

$caseCount = (Get-ChildItem -LiteralPath $destination -Directory).Count
if ($caseCount -ne 45) {
    throw "G3-B2 assembled matrix must contain 45 cases, found $caseCount"
}

$metadata = [ordered]@{
    model_protocol_version = "g3b2_two_state_v1.0"
    development_root = (Resolve-Path -LiteralPath $development).Path
    holdout_root = (Resolve-Path -LiteralPath $holdout).Path
    development_seeds = $developmentSeeds
    untouched_holdout_seeds = $holdoutSeeds
    excluded_inspected_seeds = 6..15
    case_count = $caseCount
    retry = $retry
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "assembly_metadata.json") -Encoding utf8

$manifest = Get-ChildItem -LiteralPath $destination -Recurse -File |
    Where-Object { $_.Name -ne "manifest.sha256" } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($destination.Length).TrimStart("\").Replace("\", "/")
        "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $relative
    }
$manifest | Set-Content -LiteralPath (Join-Path $destination "manifest.sha256") -Encoding utf8
Write-Host "G3B2_ASSEMBLY_OK output=$destination"
