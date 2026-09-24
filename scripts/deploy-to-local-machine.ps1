param(
  [int]$FrontendPort = 3000,
  [int]$BackendPort = 8787,
  [switch]$NoBrowser
)

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
npm run build

$backendCommand = "Set-Location '$repoRoot'; `$env:PORT='$BackendPort'; npm run dev:server"
$frontendCommand = "Set-Location '$repoRoot'; npx vite --host 127.0.0.1 --port $FrontendPort"

Write-Host "Starting backend at http://127.0.0.1:$BackendPort"
Start-Process powershell.exe -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $backendCommand)

Start-Sleep -Seconds 2

Write-Host "Starting frontend at http://127.0.0.1:$FrontendPort"
Start-Process powershell.exe -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $frontendCommand)

if (-not $NoBrowser) {
  Start-Sleep -Seconds 3
  Start-Process "http://127.0.0.1:$FrontendPort"
}

Write-Host ""
Write-Host "Local Model Lab deployment started."
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
Write-Host "Backend:  http://127.0.0.1:$BackendPort"
Write-Host ""
Write-Host "The training pipeline still creates its Python ML environment during the app's Prepare environment stage."
