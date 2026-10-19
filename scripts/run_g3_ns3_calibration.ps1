param(
    [string]$Config = "configs\g3_network_calibration.json",
    [string]$OutputRoot = "runs\g3\ns3_raw",
    [string]$BatchName = "G3B_ns3_confirmation_v1",
    [int[]]$Seeds = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot $Config
$settings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json

$baseRoot = Join-Path $projectRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $baseRoot | Out-Null
$batchName = $BatchName
$batchRoot = Join-Path $baseRoot $batchName
$retry = 0
while ($true) {
    $candidateName = Split-Path -Leaf $batchRoot
    wsl -d Ubuntu-24.04 -u tnsmsim -- test -e "/opt/tnsm-ns3/g3b-runs/$candidateName"
    $linuxExists = $LASTEXITCODE -eq 0
    if (-not (Test-Path -LiteralPath $batchRoot) -and -not $linuxExists) {
        break
    }
    $retry += 1
    $batchRoot = Join-Path $baseRoot ("{0}_retry{1:D2}" -f $batchName, $retry)
}
New-Item -ItemType Directory -Path $batchRoot | Out-Null
$linuxBatchName = Split-Path -Leaf $batchRoot
$linuxBatchRoot = "/opt/tnsm-ns3/g3b-runs/$linuxBatchName"
wsl -d Ubuntu-24.04 -u tnsmsim -- mkdir -p $linuxBatchRoot

$source = Join-Path $projectRoot "ns3\g3_80211p_calibration.cc"
$wslSource = "\\wsl.localhost\Ubuntu-24.04\opt\tnsm-ns3\ns-3.47\scratch\tnsm-80211p-calibration.cc"
Copy-Item -LiteralPath $source -Destination $wslSource -Force

wsl -d Ubuntu-24.04 -u tnsmsim -- bash -lc "cd /opt/tnsm-ns3/ns-3.47 && ./ns3 build scratch/tnsm-80211p-calibration"
if ($LASTEXITCODE -ne 0) {
    throw "ns-3 calibration program failed to build"
}

$reference = $settings.reference_simulator
$traffic = $settings.probe_traffic
$generationJitterFraction = if ($null -ne $traffic.generation_jitter_fraction) {
    $traffic.generation_jitter_fraction
} else {
    0.0
}
$runSeeds = if ($Seeds.Count -gt 0) { $Seeds } else { $settings.pilot.seeds }
$markers = @()
foreach ($regime in $settings.regimes) {
    foreach ($seed in $runSeeds) {
        $caseName = "{0}_seed{1:D4}" -f $regime.id, [int]$seed
        $caseDir = Join-Path $batchRoot $caseName
        $linuxOutput = "$linuxBatchRoot/$caseName"
        $command = @(
            "cd /opt/tnsm-ns3/ns-3.47 && ./ns3 run scratch/tnsm-80211p-calibration --",
            "--output='$linuxOutput'",
            "--protocolVersion='$($settings.protocol_version)'",
            "--regime='$($regime.id)'",
            "--seed=$seed",
            "--probeNodes=$($traffic.vehicle_count)",
            "--backgroundNodes=$($regime.background_nodes)",
            "--backgroundRate=$($regime.background_rate_hz)",
            "--backgroundRadius=$($regime.background_radius_m)",
            "--packetSize=$($traffic.packet_size_bytes)",
            "--probeRate=$($traffic.rate_hz)",
            "--generationJitterFraction=$generationJitterFraction",
            "--warmup=$($traffic.warmup_s)",
            "--measurement=$($traffic.measurement_s)",
            "--spacing=$($traffic.lateral_spacing_m)",
            "--pathLossExponent=$($reference.path_loss_exponent)",
            "--referenceLoss=$($reference.reference_loss_db)",
            "--txPower=$($reference.tx_power_dbm)",
            "--noiseFigure=$($reference.rx_noise_figure_db)"
        ) -join " "
        $marker = wsl -d Ubuntu-24.04 -u tnsmsim -- bash -lc $command
        if ($LASTEXITCODE -ne 0) {
            throw "ns-3 failed for $caseName"
        }
        $marker | Write-Host
        $markers += [pscustomobject]@{ case = $caseName; output = $marker }
        New-Item -ItemType Directory -Path $caseDir | Out-Null
        $linuxCase = "\\wsl.localhost\Ubuntu-24.04\opt\tnsm-ns3\g3b-runs\$linuxBatchName\$caseName\*"
        Copy-Item -Path $linuxCase -Destination $caseDir -Force
    }
}

$batchMetadata = [ordered]@{
    protocol_version = $settings.protocol_version
    config = (Resolve-Path -LiteralPath $configPath).Path
    config_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $configPath).Hash.ToLowerInvariant()
    source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    ns3_version = $reference.version
    retry = $retry
    cases = $markers
}
$batchMetadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batchRoot "batch_metadata.json") -Encoding utf8

$manifest = Get-ChildItem -LiteralPath $batchRoot -Recurse -File |
    Where-Object { $_.Name -ne "manifest.sha256" } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($batchRoot.Length).TrimStart("\").Replace("\", "/")
        "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $relative
    }
$manifest | Set-Content -LiteralPath (Join-Path $batchRoot "manifest.sha256") -Encoding utf8

Write-Host "G3B_NS3_BATCH_OK output=$batchRoot"
