param(
    [string]$Version = "v1.0.0",
    [string]$OutputDir = "dist/release",
    [string]$BinaryPath = "target/release/volta.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $repoRoot

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo is required"
}

if (-not (Get-Command makensis -ErrorAction SilentlyContinue)) {
    $fallbackMakensis = "C:\Program Files (x86)\NSIS\makensis.exe"
    if (Test-Path $fallbackMakensis) {
        $env:Path += ";C:\Program Files (x86)\NSIS"
    }
    else {
        throw "makensis is required (install NSIS)"
    }
}

Write-Host "[windows-installer] building release binary..."
cargo build --release

$binary = Join-Path $repoRoot $BinaryPath
if (-not (Test-Path $binary)) {
    throw "binary not found at $binary"
}

$winOut = Join-Path $repoRoot "$OutputDir/$Version/windows"
New-Item -ItemType Directory -Force -Path $winOut | Out-Null

$assetScript = Join-Path $repoRoot "packaging/windows/generate-assets.ps1"
& $assetScript

$headerBmp = Join-Path $repoRoot "packaging/windows/assets/generated/header.bmp"
$welcomeBmp = Join-Path $repoRoot "packaging/windows/assets/generated/welcome.bmp"
$pageBgBmp = Join-Path $repoRoot "packaging/windows/assets/generated/page-bg.bmp"

$nsi = Join-Path $repoRoot "packaging/windows/volta-installer.nsi"
Write-Host "[windows-installer] compiling NSIS installer..."
$makensisCmd = Get-Command makensis -ErrorAction SilentlyContinue
if ($makensisCmd) {
    & $makensisCmd.Source "/DVERSION=$Version" "/DVOLTA_BINARY=$binary" "/DOUT_DIR=$winOut" "/DHEADER_BMP=$headerBmp" "/DWELCOME_BMP=$welcomeBmp" "/DPAGE_BG_BMP=$pageBgBmp" $nsi
}
else {
    $fallbackMakensis = "C:\Program Files (x86)\NSIS\makensis.exe"
    & $fallbackMakensis "/DVERSION=$Version" "/DVOLTA_BINARY=$binary" "/DOUT_DIR=$winOut" "/DHEADER_BMP=$headerBmp" "/DWELCOME_BMP=$welcomeBmp" "/DPAGE_BG_BMP=$pageBgBmp" $nsi
}

Write-Host "[windows-installer] done: $winOut"
