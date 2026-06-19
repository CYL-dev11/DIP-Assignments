param(
    [string]$Scene = "data\chair",
    [int]$Epochs = 200,
    [int]$DebugEvery = 10,
    [string]$Python = "C:\Users\ASUS\.conda\envs\gaussian_splatting\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScenePath = Join-Path $Root $Scene
$CheckpointDir = Join-Path $ScenePath "checkpoints"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

& $Python (Join-Path $Root "train.py") `
    --colmap_dir $ScenePath `
    --checkpoint_dir $CheckpointDir `
    --num_epochs $Epochs `
    --debug_every $DebugEvery `
    --debug_samples 4
