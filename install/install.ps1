# Volta installer for Windows (PowerShell).
# Detects architecture, downloads the correct binary, installs to %LOCALAPPDATA%\volta\bin.
#
# Usage (run in PowerShell as your user — no admin required):
#   irm https://raw.githubusercontent.com/BetterCrusader/Volta/main/install/install.ps1 | iex
#
# Options (environment variables):
#   $env:VOLTA_VERSION  — version to install (default: latest)
#   $env:VOLTA_PREFIX   — install directory (default: $env:LOCALAPPDATA\volta\bin)

$ErrorActionPreference = "Stop"

$Repo    = "BetterCrusader/Volta"
$Prefix  = if ($env:VOLTA_PREFIX) { $env:VOLTA_PREFIX } else { "$env:LOCALAPPDATA\volta\bin" }

# ── Detect architecture ────────────────────────────────────────────────────────
$RawArch = (Get-WmiObject Win32_Processor | Select-Object -First 1).AddressWidth
$CpuArch = $env:PROCESSOR_ARCHITECTURE

$Arch = switch ($CpuArch) {
    "AMD64"   { "x86_64" }
    "ARM64"   { "aarch64" }
    default   { "x86_64" }   # fallback
}

# ── Resolve version ────────────────────────────────────────────────────────────
$Version = $env:VOLTA_VERSION
if (-not $Version) {
    $ApiUrl  = "https://api.github.com/repos/$Repo/releases/latest"
    $Release = Invoke-RestMethod -Uri $ApiUrl -Headers @{ "User-Agent" = "volta-installer" }
    $Version = $Release.tag_name
}

if (-not $Version) {
    Write-Error "Could not determine latest version. Set `$env:VOLTA_VERSION manually."
    exit 1
}

Write-Host "Installing Volta $Version (windows/$Arch)..."

# ── Build download URL ─────────────────────────────────────────────────────────
# Release asset naming: volta-<version>-<arch>-windows.zip
# Example: volta-v1.2.0-x86_64-windows.zip
$Asset = "volta-$Version-$Arch-windows.zip"
$Url   = "https://github.com/$Repo/releases/download/$Version/$Asset"

# ── Download ───────────────────────────────────────────────────────────────────
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) "volta-install-$([System.Guid]::NewGuid())"
New-Item -ItemType Directory -Path $Tmp | Out-Null

$ZipPath = Join-Path $Tmp $Asset
Write-Host "Downloading $Url..."
Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

# ── Extract ────────────────────────────────────────────────────────────────────
Expand-Archive -Path $ZipPath -DestinationPath $Tmp -Force

# ── Install ────────────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
Copy-Item "$Tmp\volta.exe" -Destination "$Prefix\volta.exe" -Force

# ── Cleanup ────────────────────────────────────────────────────────────────────
Remove-Item -Recurse -Force $Tmp

Write-Host ""
Write-Host "Volta installed to $Prefix\volta.exe"

# ── Add to user PATH ───────────────────────────────────────────────────────────
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$Prefix*") {
    [Environment]::SetEnvironmentVariable("PATH", "$Prefix;$UserPath", "User")
    Write-Host "Added $Prefix to your user PATH."
    Write-Host "Restart your terminal for the change to take effect."
}

Write-Host ""
Write-Host "Run: volta --version"
