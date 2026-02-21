param(
    [string]$Version = "release-v1.0.0",
    [string]$Root = "dist/release"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $repoRoot

$releaseRoot = Join-Path $repoRoot "$Root/$Version"
$windowsDir = Join-Path $releaseRoot "windows"
$macosDir = Join-Path $releaseRoot "macos"
$linuxDir = Join-Path $releaseRoot "linux"

New-Item -ItemType Directory -Force -Path $windowsDir, $macosDir, $linuxDir | Out-Null

# Always include install/uninstall scripts as release assets.
Copy-Item packaging/linux/install.sh "$linuxDir/install.sh" -Force
Copy-Item packaging/linux/uninstall.sh "$linuxDir/uninstall.sh" -Force
Copy-Item packaging/macos/install-user.sh "$macosDir/install-volta-user.sh" -Force
Copy-Item packaging/macos/uninstall.sh "$macosDir/uninstall-volta.sh" -Force

$releaseNotes = Join-Path $releaseRoot "release-notes.md"
if (-not (Test-Path $releaseNotes)) {
    $notesTemplate = @'
# Volta {VERSION}

## Highlights
- Deterministic CLI runtime and governance-gated release process.
- Cross-platform installer assets for Windows, macOS, and Linux.
- Verification-first install flow: volta version and volta doctor.

## Artifacts
- Windows: windows/VoltaSetup-<version>.exe
- macOS: macos/Volta-<version>.pkg and macos/Volta-<version>.dmg
- Linux: linux/volta-<version>-<target>.tar.gz and optional .deb
'@

    $notesTemplate.Replace("{VERSION}", $Version) | Set-Content $releaseNotes
}

$checksumFile = Join-Path $releaseRoot "checksums.txt"
if (Test-Path $checksumFile) {
    Remove-Item $checksumFile -Force
}

Get-ChildItem -Path $releaseRoot -Recurse -File |
    Where-Object { $_.Name -ne "checksums.txt" } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = Get-FileHash -Algorithm SHA256 $_.FullName
        $relative = $_.FullName.Substring($releaseRoot.Length + 1).Replace("\", "/")
        "$($hash.Hash)  $relative" | Add-Content $checksumFile
    }

Write-Host "[assemble-release] release layout prepared at $releaseRoot"
