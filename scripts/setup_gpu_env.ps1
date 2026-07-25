param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Get-RepositoryRoot {
    $root = git rev-parse --show-toplevel
    if (-not $root) {
        throw "Cannot locate git repository root."
    }
    return (Resolve-Path -LiteralPath $root).Path
}

function Assert-InRepository {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $trimChars = [char[]]@([char]92, [char]47)
    $rootTrimmed = $Root.TrimEnd($trimChars)
    $rootWithSeparator = $rootTrimmed + [System.IO.Path]::DirectorySeparatorChar
    $resolvedTrimmed = $resolved.TrimEnd($trimChars)
    $gitRoot = (Join-Path $Root ".git").TrimEnd($trimChars)
    if ($resolvedTrimmed -eq $gitRoot -or $resolvedTrimmed.StartsWith($gitRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refuse to operate on .git or paths inside .git: $resolved"
    }
    if (-not ($resolvedTrimmed -eq $rootTrimmed -or $resolvedTrimmed.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refuse to operate on path outside repository: $resolved"
    }
    return $resolved
}

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-Python312Command {
    if (Test-CommandExists "py") {
        try {
            $version = (& py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") 2>$null
            if ($version -eq "3.12") {
                return @("py", "-3.12")
            }
        } catch {
            Write-Host "py -3.12 is unavailable; checking python."
        }
    }

    if (Test-CommandExists "python") {
        $version = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if ($version -eq "3.12") {
            return @("python")
        }
    }

    throw "Python 3.12 is required. Install Python 3.12 or ensure py -3.12 / python points to 3.12."
}

function Invoke-Python312 {
    param(
        [Parameter(Mandatory = $true)][string[]]$PythonCommand,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    if ($PythonCommand.Count -eq 1) {
        & $PythonCommand[0] @Arguments
    } else {
        & $PythonCommand[0] $PythonCommand[1] @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

$root = Get-RepositoryRoot
$venv = Join-Path $root ".venv-gpu"

if ($Recreate -and (Test-Path -LiteralPath $venv)) {
    $resolvedVenv = Assert-InRepository -Root $root -Path $venv
    Write-Host "Remove old GPU virtual environment: $resolvedVenv"
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

$pythonCommand = Get-Python312Command
Invoke-Python312 $pythonCommand "-m" "venv" $venv

$venvPython = Join-Path $venv "Scripts\python.exe"
Invoke-CheckedNative -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedNative -FilePath $venvPython -Arguments @("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu128", "torch", "torchvision", "torchaudio")
Invoke-CheckedNative -FilePath $venvPython -Arguments @("-m", "pip", "install", "-r", (Join-Path $root "requirements-gpu.txt"))
Invoke-CheckedNative -FilePath $venvPython -Arguments @("-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))")
