param([string]$At = '')
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$OutDir = 'docs/results/p5_confirmation_20261123'
$StatusPath = Join-Path $OutDir 'pause_status.json'
if (-not $At) { throw 'Specify -At with an ISO time, e.g. 2026-11-25T14:00:00.' }
$target = [DateTime]::Parse($At)
$seconds = [int](($target - (Get-Date)).TotalSeconds)
$payload = @{ state='paused'; next_resume_local=$target.ToString('o');
              scheduled_script='scripts/run_p5_confirmation_scheduled.ps1' } | ConvertTo-Json -Compress
[System.IO.File]::WriteAllText((Join-Path (Get-Location) $StatusPath), $payload,
  (New-Object System.Text.UTF8Encoding($false)))
if ($seconds -gt 0) { Start-Sleep -Seconds $seconds }
Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
  '-File','D:\Research\02_研究点二_神经均衡\scripts\run_p5_confirmation_scheduled.ps1' -WindowStyle Hidden
