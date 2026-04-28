param(
    [switch]$Clean,
    [switch]$BuildInstaller
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

Set-Location $ProjectRoot

if ($Clean) {
    if (Test-Path "build") {
        Remove-Item "build" -Recurse -Force
    }
    if (Test-Path "dist") {
        Remove-Item "dist" -Recurse -Force
    }
}

Write-Host "Building demo executable..." -ForegroundColor Cyan
python scripts\check_packaging_env.py

$poetryCommand = Get-Command poetry -ErrorAction SilentlyContinue
if ($poetryCommand) {
    poetry run pyinstaller --noconfirm --clean "ml_orchestrator_demo.spec"
}
else {
    $pyinstallerCommand = Get-Command pyinstaller -ErrorAction SilentlyContinue
    if (-not $pyinstallerCommand) {
        throw "PyInstaller was not found. Install it with 'pip install pyinstaller' or use Poetry."
    }
    pyinstaller --noconfirm --clean "ml_orchestrator_demo.spec"
}

$ExePath = Join-Path $ProjectRoot "dist\\MLOrchestratorDemo\\MLOrchestratorDemo.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build completed but executable was not found at $ExePath"
}

Write-Host "Build completed: $ExePath" -ForegroundColor Green

if ($BuildInstaller) {
    & "$ScriptRoot\\build_installer.ps1"
}
