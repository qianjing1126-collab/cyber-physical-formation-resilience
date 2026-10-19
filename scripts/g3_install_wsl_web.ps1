$ErrorActionPreference = 'Continue'
$logPath = 'C:\Users\Lenovo\OneDrive\Desktop\TNSM返修\simulation_v1\g3_wsl_install.log'
"timestamp=$((Get-Date).ToString('o'))" | Set-Content -LiteralPath $logPath -Encoding UTF8
"administrator=$(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))" | Add-Content -LiteralPath $logPath -Encoding UTF8
& "$env:SystemRoot\System32\wsl.exe" --install --web-download -d Ubuntu-24.04 --no-launch 2>&1 |
    Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
"WSL_INSTALL_EXIT_CODE=$exitCode" | Tee-Object -FilePath $logPath -Append
if ($exitCode -eq 0) {
    "Installation command succeeded. A Windows restart may be required." | Tee-Object -FilePath $logPath -Append
} else {
    "Installation command failed. Keep this window open for diagnosis." | Tee-Object -FilePath $logPath -Append
}
Read-Host 'Press Enter to close this installer window'
