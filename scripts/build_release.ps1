param(
    [string]$AppVersion = "0.1.0"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

Set-Location $ProjectRoot

Write-Host "Building release package for version $AppVersion..." -ForegroundColor Cyan

& "$ScriptRoot\build_demo.ps1"
& "$ScriptRoot\build_installer.ps1" -AppVersion $AppVersion
& "$ScriptRoot\verify_release_artifact.ps1"

Write-Host "Release package completed successfully." -ForegroundColor Green
