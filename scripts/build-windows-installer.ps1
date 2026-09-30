param(
  [ValidateSet('arm64', 'x64')]
  [string]$Architecture = 'arm64'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$releaseRoot = Join-Path $repoRoot 'release'
$appDirectoryName = if ($Architecture -eq 'arm64') { 'win-arm64-unpacked' } else { 'win-unpacked' }
$appDirectory = Join-Path $releaseRoot $appDirectoryName
$payloadPath = Join-Path $releaseRoot "Local-Model-Lab-$Architecture.zip"
$installerPath = Join-Path $releaseRoot "Local Model Lab-Setup-1.0.1-$Architecture.exe"
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'

if (-not (Test-Path $csc)) {
  throw "The Windows .NET Framework C# compiler was not found at '$csc'."
}

Set-Location $repoRoot
npm run build
if ($LASTEXITCODE -ne 0) {
  throw "Frontend build failed with exit code $LASTEXITCODE."
}

$architectureFlag = if ($Architecture -eq 'arm64') { '--arm64' } else { '--x64' }
npx electron-builder --win dir $architectureFlag
if ($LASTEXITCODE -ne 0) {
  throw "Electron packaging failed with exit code $LASTEXITCODE."
}

& $csc /nologo /target:winexe /platform:anycpu /win32icon:build\icon.ico `
  /out:"$appDirectory\Uninstall Local Model Lab.exe" installer\Uninstaller.cs
if ($LASTEXITCODE -ne 0) {
  throw "Uninstaller compilation failed with exit code $LASTEXITCODE."
}

Remove-Item -LiteralPath $payloadPath -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $appDirectory '*') -DestinationPath $payloadPath -CompressionLevel Optimal

& $csc /nologo /target:winexe /platform:anycpu /win32icon:build\icon.ico `
  /reference:System.IO.Compression.dll `
  /reference:System.IO.Compression.FileSystem.dll `
  /reference:System.Windows.Forms.dll `
  /reference:Microsoft.CSharp.dll `
  /resource:"$payloadPath,ModelPayload.zip" `
  /out:"$installerPath" installer\Installer.cs
if ($LASTEXITCODE -ne 0) {
  throw "Installer compilation failed with exit code $LASTEXITCODE."
}

Write-Host "Created complete Windows installer:"
Write-Host $installerPath
