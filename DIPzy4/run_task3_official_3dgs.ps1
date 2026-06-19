param(
    [string]$Scene = "data\chair",
    [string]$GsRepo = "C:\Users\ASUS\Desktop\Stage 1_Pipeline\gaussian-splatting",
    [string]$Python = "C:\Users\ASUS\.conda\envs\gaussian_splatting\python.exe",
    [int]$Iterations = 7000,
    [switch]$Eval
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScenePath = Join-Path $Root $Scene
$OutputName = if ($Eval) { "official_3dgs_eval_output" } else { "official_3dgs_output" }
$ModelPath = Join-Path $ScenePath $OutputName

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

if (-not (Test-Path -LiteralPath (Join-Path $GsRepo "train.py"))) {
    throw "Official 3DGS repo not found. Pass -GsRepo <path-to-gaussian-splatting>."
}

$TrainArgs = @(
    (Join-Path $GsRepo "train.py"),
    "-s", $ScenePath,
    "-m", $ModelPath,
    "--iterations", $Iterations,
    "--test_iterations", $Iterations,
    "--save_iterations", $Iterations,
    "--checkpoint_iterations", $Iterations
)

if ($Eval) {
    $TrainArgs += "--eval"
}

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Official 3DGS training failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $GsRepo "render.py") -m $ModelPath --iteration $Iterations
if ($LASTEXITCODE -ne 0) {
    throw "Official 3DGS rendering failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $GsRepo "metrics.py") -m $ModelPath
if ($LASTEXITCODE -ne 0) {
    throw "Official 3DGS metrics failed with exit code $LASTEXITCODE"
}
