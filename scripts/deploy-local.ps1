param(
  [int]$FrontendPort = 3000,
  [int]$BackendPort = 8787
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

npm run install:all
npm run build

Write-Host "Start backend with: $env:PORT=$BackendPort; npm run dev:server"
Write-Host "Start frontend preview with: npm run preview -- --host 0.0.0.0 --port $FrontendPort"
