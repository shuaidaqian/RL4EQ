$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$Python = '.venv-gpu\Scripts\python.exe'
$Experiment = 'soft_data_variance_pilot_10x1024_20261122'
$Seeds = 240002,240003,240004,240005,240006,240007,240008,240009,240010,240011
$Manifest = 'docs\protocols\continual_v2_soft_data_variance_pilot_10x1024_20261122.json'
$Output = 'docs\results\continual_v2_soft_data_variance_pilot_20261122\variance_pilot.json'
$OutputDir = 'docs\results\continual_v2_soft_data_variance_pilot_20261122'
$ExperimentSeconds = 78000
$ReserveSeconds = 3600
$Started = [DateTimeOffset]::UtcNow
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path $OutputDir $Path), $Text, $encoding)
}
Write-Utf8NoBom 'run_started_utc.txt' $Started.ToString('o')
try {
    $Elapsed = [int]([DateTimeOffset]::UtcNow - $Started).TotalSeconds
    $Remaining = $ExperimentSeconds - $Elapsed
    if ($Remaining -le 0) { throw 'No runtime remains within the pilot budget.' }
    $Duration = '{0}s' -f $Remaining
    & timeout --foreground -k 30s $Duration $Python scripts/compare_continual_variants.py `
        --manifest $Manifest --seeds $Seeds --frames 1024 `
        --experiment $Experiment --output $Output --device cuda `
        --stream-rows --checkpoint-interval 128
    $RunCode = $LASTEXITCODE
    $Now = [DateTimeOffset]::UtcNow
    $ElapsedSeconds = [int]($Now - $Started).TotalSeconds
    if ($RunCode -eq 124 -or $RunCode -eq 137) { $State = 'incomplete_time_limit' }
    elseif ($RunCode -eq 0) { $State = 'completed' }
    else { $State = 'failed' }
    [PSCustomObject]@{
      experiment=$Experiment; status=$State; process_exit_code=$RunCode
      started_utc=$Started.ToString('o'); finished_utc=$Now.ToString('o')
      elapsed_seconds=$ElapsedSeconds; hard_total_wall_limit_seconds=81600
      seeds=$Seeds; frames=1024; snrs=@(0,5,10,15); device='cuda:0'
      output=$Output; timeout_exit_codes=@(124,137)
      note='Variance-only pilot; no efficacy inference; no extension or seed replacement.'
    } | ConvertTo-Json -Depth 5 | ForEach-Object { Write-Utf8NoBom 'execution_status.json' $_ }
    if ($State -ne 'completed') { throw "Pilot ended as $State (exit $RunCode); do not resume or replace seeds." }
}
catch {
    [PSCustomObject]@{
      experiment=$Experiment; status='failed_or_stopped'; error=$_.Exception.Message
      checked_utc=[DateTimeOffset]::UtcNow.ToString('o')
      note='Preserve reservation and partial output; do not rerun or replace seeds.'
    } | ConvertTo-Json -Depth 4 | ForEach-Object { Write-Utf8NoBom 'execution_error.json' $_ }
    throw
}
