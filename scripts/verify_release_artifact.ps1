param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$InstallerPath,
    [string]$DemoExePath
)

$ErrorActionPreference = "Stop"

if (-not $InstallerPath) {
    $InstallerPath = Join-Path $ProjectRoot "dist\installer\MLOrchestratorDemoSetup.exe"
}

if (-not $DemoExePath) {
    $DemoExePath = Join-Path $ProjectRoot "dist\MLOrchestratorDemo\MLOrchestratorDemo.exe"
}

if (-not (Test-Path $DemoExePath)) {
    throw "Demo executable not found at $DemoExePath"
}

if (-not (Test-Path $InstallerPath)) {
    throw "Installer not found at $InstallerPath"
}

$demoItem = Get-Item $DemoExePath
$installerItem = Get-Item $InstallerPath

if ($demoItem.Length -le 0) {
    throw "Demo executable is empty: $DemoExePath"
}

if ($installerItem.Length -le 0) {
    throw "Installer is empty: $InstallerPath"
}

$installerHash = Get-FileHash -Algorithm SHA256 -Path $InstallerPath
$checksumPath = Join-Path $installerItem.DirectoryName "MLOrchestratorDemoSetup.exe.sha256.txt"
$checksumLine = "{0}  {1}" -f $installerHash.Hash, $installerItem.Name
Set-Content -Path $checksumPath -Value $checksumLine -Encoding ascii

Write-Host "Verified release artifacts:" -ForegroundColor Green
Write-Host "  Demo executable: $DemoExePath ($([math]::Round($demoItem.Length / 1MB, 2)) MB)"
Write-Host "  Installer:       $InstallerPath ($([math]::Round($installerItem.Length / 1MB, 2)) MB)"
Write-Host "  SHA256:          $($installerHash.Hash)"
Write-Host "  Checksum file:    $checksumPath"
