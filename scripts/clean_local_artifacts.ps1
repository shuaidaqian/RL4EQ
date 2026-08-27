param()

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
        throw "Refuse to clean .git or paths inside .git: $resolved"
    }
    if (-not ($resolvedTrimmed -eq $rootTrimmed -or $resolvedTrimmed.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refuse to clean path outside repository: $resolved"
    }
    return $resolved
}

function Remove-GuardedPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "Skip missing path: $Path"
        return
    }
    $resolved = Assert-InRepository -Root $Root -Path $Path
    if ((Split-Path -Leaf $resolved) -eq ".git") {
        Write-Host "Skip .git: $resolved"
        return
    }
    Write-Host "Clean: $resolved"
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$root = Get-RepositoryRoot
$exactNames = @(".git_original", ".pytest_cache", ".agents", "logs", "pretrained", "artifacts")
foreach ($name in $exactNames) {
    Remove-GuardedPath -Root $root -Path (Join-Path $root $name)
}

Get-ChildItem -LiteralPath $root -Directory -Recurse -Force -Filter "__pycache__" |
    Where-Object {
        $trimChars = [char[]]@([char]92, [char]47)
        $gitRoot = (Join-Path $root ".git").TrimEnd($trimChars)
        $candidate = $_.FullName.TrimEnd($trimChars)
        -not ($candidate -eq $gitRoot -or $candidate.StartsWith($gitRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase))
    } |
    ForEach-Object {
        Remove-GuardedPath -Root $root -Path $_.FullName
    }
