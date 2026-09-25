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

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

Write-Host "Deploying Local Model Lab to this machine from: $repoRoot"

Assert-Command -Name 'node' -InstallHint 'Install Node.js 20+ from https://nodejs.org/'
Assert-Command -Name 'npm' -InstallHint 'Install Node.js 20+ from https://nodejs.org/'
Assert-Command -Name 'python' -InstallHint 'Install Python 3.11 and ensure python.exe is on PATH.'

npm run install:all
npm run desktop:package

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
  Start-Process -FilePath $launcher.FullName
}
