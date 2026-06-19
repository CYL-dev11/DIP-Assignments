param(
    [string]$Scene = "data\chair",
    [string]$Python = "C:\Users\ASUS\.conda\envs\gaussian_splatting\python.exe",
    [string]$ColmapBin = "C:\Users\ASUS\Downloads\colmap-x64-windows-cuda\bin"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScenePath = Join-Path $Root $Scene

if (Test-Path -LiteralPath $ColmapBin) {
    $env:PATH = "$ColmapBin;$env:PATH"
}

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

function Invoke-Step {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step { & $Python (Join-Path $Root "mvs_with_colmap.py") --data_dir $ScenePath }
Invoke-Step { & $Python (Join-Path $Root "debug_mvs_by_projecting_pts.py") --data_dir $ScenePath }
