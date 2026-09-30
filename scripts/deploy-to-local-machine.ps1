param([switch]$BuildOnly)

$ErrorActionPreference = 'Stop'

function Assert-Command {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string]$InstallHint
  )

  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command '$Name' was not found. $InstallHint"
  }
}

function Invoke-CheckedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments,
    [int]$MaxAttempts = 1
  )

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    & $Name @Arguments
    if ($LASTEXITCODE -eq 0) {
      return
    }
    if ($attempt -lt $MaxAttempts) {
      Write-Host "Command failed with exit code $LASTEXITCODE. Retrying in 3 seconds..."
      Start-Sleep -Seconds 3
    }
  }
  throw "Command failed after $MaxAttempts attempt(s) with exit code ${LASTEXITCODE}: $Name $($Arguments -join ' ')"
}

function Stop-ReleaseProcesses {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseRoot
  )

  $normalizedReleaseRoot = [IO.Path]::GetFullPath($ReleaseRoot)
  $runningProcesses = @(
    Get-Process | Where-Object {
      $_.Path -and
        [IO.Path]::GetFullPath($_.Path).StartsWith(
          $normalizedReleaseRoot,
          [StringComparison]::OrdinalIgnoreCase
        )
    }
  )

  if ($runningProcesses.Count -eq 0) {
    return
  }

  Write-Host "Closing $($runningProcesses.Count) running Local Model Lab process(es) before packaging..."
  foreach ($process in $runningProcesses) {
    Stop-Process -Id $process.Id -Force -ErrorAction Stop
  }

  $remainingProcessIds = @($runningProcesses | ForEach-Object { $_.Id })
  for ($attempt = 1; $attempt -le 20 -and $remainingProcessIds.Count -gt 0; $attempt++) {
    Start-Sleep -Milliseconds 500
    $remainingProcessIds = @(
      foreach ($processId in $remainingProcessIds) {
        if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
          $processId
        }
      }
    )
  }
  if ($remainingProcessIds.Count -gt 0) {
    throw "Could not close Local Model Lab process ID(s): $($remainingProcessIds -join ', ')"
  }
}

function Test-RepositoryRoot {
  param([string]$Candidate)

  return $Candidate -and
    (Test-Path -LiteralPath (Join-Path $Candidate 'package.json') -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $Candidate 'server\package.json') -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $Candidate 'electron\main.js') -PathType Leaf)
}

function Find-RepositoryRoot {
  if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptDirectory = [IO.Path]::GetFullPath($PSScriptRoot)
    foreach ($candidate in @($scriptDirectory, (Split-Path -Parent $scriptDirectory))) {
      if (Test-RepositoryRoot -Candidate $candidate) {
        return $candidate
      }
    }
  }

  $candidate = (Get-Location).Path
  while ($candidate) {
    if (Test-RepositoryRoot -Candidate $candidate) {
      return $candidate
    }
    $parent = Split-Path -Parent $candidate
    if (-not $parent -or $parent -eq $candidate) {
      break
    }
    $candidate = $parent
  }

  return $null
}

function Get-RepositoryRoot {
  $localRepository = Find-RepositoryRoot
  if ($localRepository) {
    return $localRepository
  }

  Assert-Command -Name 'git' -InstallHint 'Install Git from https://git-scm.com/download/win'
  $sourceRoot = Join-Path $env:LOCALAPPDATA 'Local Model Lab\source'
  if (Test-RepositoryRoot -Candidate $sourceRoot) {
    Write-Host "Updating the existing Local Model Lab source at: $sourceRoot"
    Invoke-CheckedCommand -Name 'git' -Arguments @('-C', $sourceRoot, 'pull', '--ff-only', 'origin', 'main')
    return $sourceRoot
  }

  if (Test-Path -LiteralPath $sourceRoot) {
    throw "The installation source path exists but is not a valid Model Trainer checkout: $sourceRoot"
  }

  $sourceParent = Split-Path -Parent $sourceRoot
  New-Item -ItemType Directory -Path $sourceParent -Force | Out-Null
  Write-Host "No local checkout was found. Cloning Model Trainer to: $sourceRoot"
  Invoke-CheckedCommand -Name 'git' -Arguments @(
    'clone',
    'https://github.com/jwgmpi-wolff/modeltrainer.git',
    $sourceRoot
  )
  return $sourceRoot
}

Assert-Command -Name 'node' -InstallHint 'Install Node.js 20+ from https://nodejs.org/'
Assert-Command -Name 'npm' -InstallHint 'Install Node.js 20+ from https://nodejs.org/'

$repoRoot = Get-RepositoryRoot
Set-Location $repoRoot
Write-Host "Deploying Local Model Lab to this machine from: $repoRoot"

Invoke-CheckedCommand -Name 'npm' -Arguments @('run', 'install:all')
Stop-ReleaseProcesses -ReleaseRoot (Join-Path $repoRoot 'release')
Invoke-CheckedCommand -Name 'npm' -Arguments @('run', 'desktop:package') -MaxAttempts 3

$launcher = Get-ChildItem -Path (Join-Path $repoRoot 'release') -Filter 'Local Model Lab.exe' -Recurse |
  Where-Object { $_.FullName -match 'win-.+-unpacked' } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $launcher) {
  throw "Desktop packaging finished but no unpacked launcher was found under '$repoRoot\release'."
}

$shell = New-Object -ComObject WScript.Shell
$desktopShortcut = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Local Model Lab.lnk'
$startMenuDir = Join-Path ([Environment]::GetFolderPath('Programs')) 'Local Model Lab'
$startMenuShortcut = Join-Path $startMenuDir 'Local Model Lab.lnk'
New-Item -ItemType Directory -Path $startMenuDir -Force | Out-Null

foreach ($shortcutPath in @($desktopShortcut, $startMenuShortcut)) {
  $shortcut = $shell.CreateShortcut($shortcutPath)
  $shortcut.TargetPath = $launcher.FullName
  $shortcut.WorkingDirectory = $launcher.DirectoryName
  $shortcut.Description = 'Launch Local Model Lab'
  $shortcut.Save()
}

Write-Host ""
Write-Host "Local Model Lab desktop launcher created:"
Write-Host $launcher.FullName
Write-Host "Desktop shortcut:"
Write-Host $desktopShortcut
Write-Host "Start menu shortcut:"
Write-Host $startMenuShortcut
Write-Host "Launch Local Model Lab by double-clicking either shortcut."

if (-not $BuildOnly) {
  $appProcess = Start-Process -FilePath $launcher.FullName -WorkingDirectory $launcher.DirectoryName -PassThru
  Start-Sleep -Seconds 3
  if ($appProcess.HasExited) {
    throw "Local Model Lab exited immediately with code $($appProcess.ExitCode). Run '$($launcher.FullName) --enable-logging' to inspect startup output."
  }
  $null = $shell.AppActivate($appProcess.Id)
  Write-Host "Local Model Lab is running as process $($appProcess.Id)."
}
