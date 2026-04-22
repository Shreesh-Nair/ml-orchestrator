param(
    [string]$AppVersion = "0.1.0"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
$InstallerScript = Join-Path $ProjectRoot "installer\\ml_orchestrator_demo.iss"
$DistDir = Join-Path $ProjectRoot "dist\\MLOrchestratorDemo"

if (-not (Test-Path $DistDir)) {
    throw "Executable directory not found at $DistDir. Run scripts\\build_demo.ps1 first."
}

$IsccPath = $null
$candidatePaths = @(
    "${env:ProgramFiles(x86)}\\Inno Setup 6\\ISCC.exe",
    "${env:ProgramFiles}\\Inno Setup 6\\ISCC.exe"
)

foreach ($candidate in $candidatePaths) {
    if (Test-Path $candidate) {
        $IsccPath = $candidate
        break
    }
}

if (-not $IsccPath) {
    $cmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($cmd) {
        $IsccPath = $cmd.Source
    }
}

if (-not $IsccPath) {
    throw "Inno Setup (ISCC.exe) not found. Install Inno Setup 6, then re-run this script."
}

Write-Host "Building installer..." -ForegroundColor Cyan
& $IsccPath "/DAppVersion=$AppVersion" $InstallerScript

Write-Host "Installer build completed in dist\\installer" -ForegroundColor Green
