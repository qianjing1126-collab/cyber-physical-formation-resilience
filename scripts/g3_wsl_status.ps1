$ErrorActionPreference = 'Continue'
$outputPath = 'C:\Users\Lenovo\OneDrive\Desktop\TNSM返修\simulation_v1\g3_wsl_status.log'
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("timestamp=$((Get-Date).ToString('o'))")
$lines.Add("administrator=$(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))")
foreach ($featureName in @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')) {
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
    $lines.Add("feature:$featureName=$($feature.State)")
}
$lines.Add('wsl_status_begin')
$lines.AddRange([string[]](& "$env:SystemRoot\System32\wsl.exe" --status 2>&1))
$lines.Add("wsl_status_exit=$LASTEXITCODE")
$lines.Add('wsl_list_begin')
$lines.AddRange([string[]](& "$env:SystemRoot\System32\wsl.exe" --list --verbose 2>&1))
$lines.Add("wsl_list_exit=$LASTEXITCODE")
$lines | Set-Content -LiteralPath $outputPath -Encoding UTF8
