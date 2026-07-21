$ErrorActionPreference = "Stop"

$taskName = "Codex - Monitor ibm_fez calibration"
$artifactDirectory = Join-Path $PSScriptRoot "fire_opal_pbmc68k_q60_modules_b4"
$latestPath = Join-Path $artifactDirectory "ibm_fez_calibration_latest.json"
$alertPath = Join-Path $artifactDirectory "ibm_fez_calibration_alert_latest.json"

$arguments = @(
    "-d",
    "Ubuntu-Preview-C",
    "--",
    "bash",
    "-lc",
    "cd /mnt/c/Users/Lenna/SynologyDrive/qlab/ML_adv && /home/bram/.venvs/qiskit/bin/python monitor_ibm_fez_calibration.py"
)

& "$env:SystemRoot\System32\wsl.exe" @arguments
$monitorExitCode = $LASTEXITCODE

if ($monitorExitCode -ne 0) {
    Disable-ScheduledTask -TaskName $taskName | Out-Null
    exit $monitorExitCode
}

if ((Test-Path -LiteralPath $latestPath) -and (Test-Path -LiteralPath $alertPath)) {
    $latest = Get-Content -Raw -LiteralPath $latestPath | ConvertFrom-Json
    $alert = Get-Content -Raw -LiteralPath $alertPath | ConvertFrom-Json
    $terminalStatuses = @(
        "new_calibration_validated",
        "validate_only_failed_no_retry_for_this_calibration"
    )
    $sameCalibration = (
        $latest.calibration_changed_since_previous_check -eq $true -and
        $alert.calibration_last_update_utc -eq $latest.calibration_last_update_utc
    )
    if ($sameCalibration -and $terminalStatuses -contains $alert.status) {
        Disable-ScheduledTask -TaskName $taskName | Out-Null
    }
}

exit 0
