<#
.SYNOPSIS
    Task runner for Windows: the same tasks as the Makefile, without needing `make`.

.DESCRIPTION
    Run through the wrapper (no execution-policy changes needed), from the repository root:
        scripts\run.cmd <task> [-Config <path>]

    Examples:
        scripts\run.cmd help
        scripts\run.cmd data
        scripts\run.cmd train -Config configs/experiments/e1_small_cnn_subset.yaml
        scripts\run.cmd airflow-docker-up
        scripts\run.cmd airflow-docker-run -Config configs/experiments/e1_small_cnn_subset.yaml
        scripts\run.cmd redact -Text "Hi, I'm Joseph Zarecki, call me at 727-360-0442"
        scripts\run.cmd redact -Model e2 -WordSnap -Text "..."
        scripts\run.cmd train-gpu -Config configs/experiments/e2_dilated_cnn.yaml
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $Task = "help",
    [string] $Config = "configs/default.yaml",
    [string] $Text,
    [string] $Model,
    [switch] $WordSnap
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
if (-not $env:TF_CPP_MIN_LOG_LEVEL) { $env:TF_CPP_MIN_LOG_LEVEL = "2" }

function Invoke-Native {
    param([string] $Exe, [string[]] $Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed (exit code $LASTEXITCODE)" }
}

function Invoke-Pipeline {
    param([string[]] $Steps, [string] $Cfg = $Config)
    Invoke-Native "uv" (@("run", "python", "-m", "pipelines.cli", "--config", $Cfg) + $Steps)
}

function Get-GitLineage {
    $commit = (git rev-parse --short HEAD 2>$null)
    if (-not $commit) { $commit = "unknown" }
    $status = (git status --porcelain --untracked-files=no 2>$null)
    $dirty = if ($status) { "true" } else { "false" }
    return @{ Commit = $commit; Dirty = $dirty }
}

# Name -> (description, action). Keep in sync with the Makefile.
$Tasks = [ordered]@{
    "help"                = @("List tasks", { Show-Help })
    "setup"               = @("Create .venv and install locked dependencies (uv sync)", {
            Invoke-Native "uv" @("sync") })
    "lint"                = @("Ruff lint + format check (same as CI)", {
            Invoke-Native "uv" @("run", "ruff", "check", ".")
            Invoke-Native "uv" @("run", "ruff", "format", "--check", ".") })
    "format"              = @("Auto-format code", {
            Invoke-Native "uv" @("run", "ruff", "format", ".")
            Invoke-Native "uv" @("run", "ruff", "check", "--fix", ".") })
    "test"                = @("Unit, integration and smoke tests", {
            Invoke-Native "uv" @("run", "pytest") })
    "data"                = @("Download, validate and split the data", {
            Invoke-Pipeline @("ingest", "validate_raw", "preprocess", "validate_processed") })
    "validate"            = @("Re-run data validation", {
            Invoke-Pipeline @("validate_raw", "validate_processed") })
    "baseline"            = @("Evaluate the regex baseline (logs to MLflow)", {
            Invoke-Pipeline @("evaluate_baseline") })
    "train"               = @("Train + evaluate one Keras model (-Config configs/experiments/<name>.yaml)", {
            Invoke-Pipeline @("train") })
    "experiments"         = @("Train every config in configs/experiments/", {
            Get-ChildItem "configs/experiments/*.yaml" | Sort-Object Name | ForEach-Object {
                Write-Host "=== $($_.Name) ===" -ForegroundColor Cyan
                Invoke-Pipeline @("train") ("configs/experiments/" + $_.Name)
            } })
    "pipeline"            = @("Every step without Airflow (trains -Config; default config is slow)", {
            Invoke-Pipeline @("all") })
    "status"              = @("Progress of running/interrupted trainings (epoch, ETA)", {
            Invoke-Native "uv" @("run", "python", "-m", "pipelines.status") })
    "decode-sweep"        = @("Recall/precision curve for the latest model of -Config", {
            Invoke-Pipeline @("decode_sweep") })
    "compare"             = @("Write reports/experiments/comparison.md (all models side by side)", {
            Invoke-Pipeline @("compare") })
    "redact"              = @("Redact -Text with a trained model (-Model e2, -WordSnap optional)", {
            if (-not $Text) { throw "pass the text to redact: scripts\run.cmd redact -Text `"...`"" }
            $redactArgs = @("run", "python", "-m", "src.serving.redactor")
            if ($Model) { $redactArgs += @("--model", $Model) }
            if ($WordSnap) { $redactArgs += "--word-snap" }
            Invoke-Native "uv" ($redactArgs + @($Text)) })
    "hooks"               = @("Install the git pre-commit hooks (once per clone)", {
            Invoke-Native "uv" @("run", "pre-commit", "install") })
    "train-gpu"           = @("Train -Config on the NVIDIA GPU (Docker); see docs/HOW_TO_RUN.md", {
            Invoke-Native "docker" @("compose", "--profile", "gpu", "run", "--rm", "--build",
                "trainer-gpu", "train", "--config", $Config) })
    "mlflow"              = @("MLflow UI for local runs on http://localhost:5000", {
            Invoke-Native "uv" @("run", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5000") })
    "docker-build"        = @("Build the pipeline image (records the git commit)", {
            $git = Get-GitLineage
            Invoke-Native "docker" @("build", "-t", "pii-redaction",
                "--build-arg", "GIT_COMMIT=$($git.Commit)", "--build-arg", "GIT_DIRTY=$($git.Dirty)", ".") })
    "docker-train"        = @("Train -Config inside the pipeline image", {
            Invoke-Native "docker" @("run", "--rm",
                "-v", "$Root\data:/app/data", "-v", "$Root\reports:/app/reports", "-v", "$Root\artifacts:/app/artifacts",
                "pii-redaction", "python", "-m", "pipelines.cli", "train", "--config", $Config) })
    "airflow-docker-up"   = @("Start Airflow (http://localhost:8080) + its MLflow UI (:5001) in Docker", {
            Invoke-Native "docker" @("compose", "up", "-d", "--build", "airflow")
            Write-Host "Airflow: http://localhost:8080   MLflow (Docker runs): http://localhost:5001" -ForegroundColor Green
            Write-Host "Allow ~30 s for Airflow to start." })
    "airflow-docker-run"  = @("Trigger one DAG run in Docker Airflow and wait for it (-Config ...)", {
            Invoke-Native "docker" @("compose", "exec", "airflow", "airflow-run", $Config) })
    "airflow-docker-logs" = @("Follow the Docker Airflow logs (Ctrl+C to stop following)", {
            Invoke-Native "docker" @("compose", "logs", "-f", "airflow") })
    "airflow-docker-down" = @("Stop Docker Airflow (state is kept in a Docker volume)", {
            Invoke-Native "docker" @("compose", "down") })
    "airflow"             = @("Native Airflow: not supported on Windows (use airflow-docker-up)", {
            throw "Airflow cannot run natively on Windows. Use: scripts\run.cmd airflow-docker-up" })
    "airflow-run"         = @("Native Airflow: not supported on Windows (use airflow-docker-run)", {
            throw "Airflow cannot run natively on Windows. Use: scripts\run.cmd airflow-docker-run -Config <path>" })
}

function Show-Help {
    Write-Host "Usage: scripts\run.cmd <task> [-Config <path>]`n"
    foreach ($name in $Tasks.Keys) { Write-Host ("  {0,-20} {1}" -f $name, $Tasks[$name][0]) }
}

if (-not $Tasks.Contains($Task)) {
    Write-Host "Unknown task '$Task'." -ForegroundColor Red
    Show-Help
    exit 2
}
try {
    & $Tasks[$Task][1]
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
