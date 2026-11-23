$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$Python = '.venv-gpu\Scripts\python.exe'
$Manifest = 'docs\protocols\p5_confirmation_manifest_draft_20261123.json'
$OutDir = 'docs/results/p5_confirmation_20261123'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir 'run_scheduled.log'
$StatusPath = Join-Path $OutDir 'scheduled_status.json'
$Timeout = 'C:\Program Files\Git\usr\bin\timeout.exe'
$WindowStartHour = 8
$PauseBufferSeconds = 120

$Runs = @(
  @{ Snr='0'; Name='p5_confirmation_snr0_20261123'; Seeds=@(250005,250006,250008,250009,250012,250014,250016,250020,250022,250023,250024,250026,250030,250033,250034,250041,250044,250046,250048,250049,250053,250054,250057,250060,250061,250067,250074,250075,250077,250079,250085,250087,250089,250090,250094,250097,250099,250100,250101,250102,250106,250107,250108,250110,250111,250114,250116,250118,250122,250123,250126,250127,250131,250132,250133,250135,250136,250140,250146,250150,250152,250155,250158,250161,250166,250169,250170,250176,250177,250185,250187,250190,250191,250192,250195,250202,250204,250209,250212,250213,250221,250224,250228,250232,250236,250238,250239,250242,250243,250245,250249,250250,250257,250258,250260,250261,250263,250265,250266,250274,250276,250279,250281,250282,250284,250288,250289,250292,250294,250295,250300,250301,250303,250307,250312,250315,250322,250326,250327,250329,250331,250332,250333,250339,250340,250341,250344,250347,250348,250352,250356,250359,250361,250363,250367,250368,250370,250371,250372,250374,250376,250377,250379,250380,250382,250384,250394,250395,250396,250401,250404,250410,250416,250418,250421,250424,250426,250427,250430,250432,250438,250443,250445,250450,250451,250453,250457,250458,250459,250460,250463,250464,250467,250470,250471,250472,250475,250476,250477,250482,250483,250485,250486,250488,250489,250494,250495,250496,250497,250498,250503,250504,250506,250510,250515,250516,250517,250520,250523,250524,250526,250528,250532,250534,250535,250542,250545,250546,250547,250549,250550,250551,250552,250553,250558,250559,250561,250562,250564,250565,250567,250569,250573,250575,250578,250581,250586,250593,250598,250602,250604,250606,250608,250613,250617,250618,250624,250626,250629,250632,250633,250634,250636,250637,250639,250645,250646,250647,250655,250658,250660,250665,250667,250668,250669,250672,250674,250679,250681,250683,250684,250686,250687,250691,250692,250693,250698,250699,250700); Output='docs/results/p5_confirmation_20261123/snr0.json' },
  @{ Snr='5'; Name='p5_confirmation_snr5_20261123'; Seeds=@(250001,250007,250010,250039,250040,250043,250045,250063,250065,250069,250088,250098,250104,250109,250112,250125,250128,250143,250149,250162,250163,250164,250167,250175,250179,250183,250193,250198,250200,250207,250216,250222,250229,250244,250267,250268,250310,250313,250330,250350,250354,250373,250422,250431,250435,250437,250442,250454,250466,250469,250473,250478,250487,250491,250507,250525,250556,250572,250582,250591,250592,250623,250631,250640,250650,250677,250688,250689,250697); Output='docs/results/p5_confirmation_20261123/snr5.json' },
  @{ Snr='10'; Name='p5_confirmation_snr10_20261123'; Seeds=@(250002,250013,250018,250028,250047,250064,250068,250073,250078,250082,250096,250105,250124,250153,250156,250157,250172,250178,250186,250189,250196,250225,250234,250237,250240,250241,250247,250252,250255,250256,250269,250270,250286,250342,250343,250346,250364,250385,250386,250392,250393,250425,250429,250447,250479,250493,250505,250508,250518,250531,250536,250548,250554,250570,250571,250576,250577,250580,250583,250585,250600,250601,250605,250609,250614,250625,250628,250642,250652,250653,250659,250675,250682); Output='docs/results/p5_confirmation_20261123/snr10.json' },
  @{ Snr='15'; Name='p5_confirmation_snr15_20261123'; Seeds=@(250011,250017,250019,250027,250029,250032,250037,250042,250050,250051,250055,250056,250058,250062,250066,250071,250080,250095,250113,250115,250121,250129,250141,250144,250145,250151,250160,250171,250174,250182,250184,250197,250199,250203,250205,250214,250218,250220,250223,250226,250227,250230,250231,250233,250253,250254,250262,250271,250273,250275,250277,250278,250299,250305,250308,250316,250319,250320,250323,250324,250349,250355,250357,250360,250365,250369,250387,250389,250390,250391,250397,250402,250409,250415,250428,250436,250448,250455,250456,250461,250474,250481,250484,250490,250500,250502,250509,250512,250513,250514,250519,250521,250522,250527,250537,250539,250555,250557,250560,250566,250587,250590,250595,250603,250611,250619,250627,250630,250635,250641,250643,250651,250654,250656,250663,250671); Output='docs/results/p5_confirmation_20261123/snr15.json' }
)

function Write-Status([string]$State, [string]$Name, [string]$NextResume) {
  $payload = @{ state=$State; experiment=$Name; next_resume_local=$NextResume;
                updated_local=(Get-Date).ToString('o') } | ConvertTo-Json -Compress
  [System.IO.File]::WriteAllText((Join-Path (Get-Location) $StatusPath), $payload,
    (New-Object System.Text.UTF8Encoding($false)))
}

function Seconds-Until([DateTime]$Target) {
  $value = [int](($Target - (Get-Date)).TotalSeconds)
  if ($value -lt 0) { return 0 }
  return $value
}

try {
  while ($true) {
    $Run = $null
    foreach ($Candidate in $Runs) {
      $state = $null
      if (Test-Path $Candidate.Output) {
        try { $state = (Get-Content -Raw $Candidate.Output | ConvertFrom-Json).status } catch { $state = $null }
      }
      if ($state -eq 'completed') {
        & $Python scripts/finalize_registry.py --experiment $Candidate.Name --result $Candidate.Output *>> $Log
        if ($LASTEXITCODE -ne 0) { throw "registry finalize failed: $($Candidate.Name)" }
        continue
      }
      $Run = $Candidate
      $State = $state
      break
    }
    if ($null -eq $Run) {
      Write-Status 'completed' 'all' ''
      Add-Content $Log "completed all $((Get-Date).ToString('o'))"
      break
    }
    $now = Get-Date
    if ($now.Hour -lt $WindowStartHour) {
      $target = [DateTime]::Today.AddHours($WindowStartHour)
      Write-Status 'waiting_window' $Run.Name $target.ToString('o')
      Add-Content $Log "outside window; sleeping until $($target.ToString('o'))"
      Start-Sleep -Seconds (Seconds-Until $target)
      continue
    }
    $midnight = [DateTime]::Today.AddDays(1)
    $seconds = (Seconds-Until $midnight) - $PauseBufferSeconds
    if ($seconds -lt 600) {
      $target = $midnight.AddHours($WindowStartHour)
      Write-Status 'waiting_window' $Run.Name $target.ToString('o')
      Add-Content $Log "window nearly over; sleeping until $($target.ToString('o'))"
      Start-Sleep -Seconds (Seconds-Until $target)
      continue
    }
    if ($State -eq 'interrupted') {
      $payload = Get-Content -Raw $Run.Output | ConvertFrom-Json
      $payload.status = 'failed'
      $payload | ConvertTo-Json -Depth 30 | Set-Content -Path $Run.Output -Encoding UTF8
      $State = 'failed'
    }
    $CliArgs = @('scripts/compare_continual_variants.py', '--manifest', $Manifest, '--seeds') + $Run.Seeds + @(
      '--frames', '11719', '--snrs', $Run.Snr, '--experiment', $Run.Name, '--output', $Run.Output,
      '--device', 'cuda', '--stream-rows', '--checkpoint-interval', '512',
      '--accel', 'graph', '--minimal-diagnostics', '--purpose', 'confirmation')
    if ($State -in @('running', 'failed')) { $CliArgs += '--resume' }
    Write-Status 'running' $Run.Name ''
    Add-Content $Log "start $($Run.Name) previous_state=$State window_seconds=$seconds $((Get-Date).ToString('o'))"
    & $Timeout --foreground -k 60s $seconds $Python @CliArgs *>> $Log
    $Code = $LASTEXITCODE
    Add-Content $Log "exit $($Run.Name) code=$Code $((Get-Date).ToString('o'))"
    if ($Code -eq 0) {
      & $Python scripts/finalize_registry.py --experiment $Run.Name --result $Run.Output *>> $Log
      if ($LASTEXITCODE -ne 0) { throw "registry finalize failed: $($Run.Name)" }
      continue
    }
    if ($Code -eq 124 -or $Code -eq 137) {
      Add-Content $Log "window timeout; will resume after $WindowStartHour:00"
      continue
    }
    throw "P5 confirmation stopped at $($Run.Name), exit $Code; preserve results, do not auto-rerun."
  }
} catch {
  Write-Status 'failed' 'supervisor' ''
  Add-Content $Log "SUPERVISOR FAILED: $($_.Exception.Message)"
  throw
}
