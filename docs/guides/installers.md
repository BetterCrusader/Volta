# Volta Installer System (Windows, macOS, Linux)

This guide defines the production installer stack for Volta CLI.

## Platform Strategy

- Windows (priority): NSIS-based custom installer (`VoltaSetup-<version>.exe`) with branded copy, optional PATH setup, optional `doctor` verification, and uninstaller.
- macOS: dual path
  - managed/admin flow: signed-ready `.pkg` (optional `.dmg` wrapper), installs to `/usr/local/lib/volta/bin` and links `/usr/local/bin/volta`
  - no-admin flow: user installer script to `~/.volta/bin`
- Linux: tarball + install/uninstall scripts (user or system mode), optional `.deb` package.

## Directory Layout

```text
packaging/
  windows/
    volta-installer.nsi
    design-tokens.json
    assets/
  macos/
    build-pkg.sh
    install-user.sh
    uninstall.sh
    scripts/{preinstall,postinstall}
  linux/
    install.sh
    uninstall.sh
    build-tarball.sh
    build-deb.sh
    deb/{postinst,prerm}

scripts/installer/
  build-windows-installer.ps1
  assemble-release.ps1
  assemble-release.sh
```

## Build Commands

### 1) Build Volta binary

```bash
cargo build --release
```

### 2) Windows installer

```powershell
pwsh ./scripts/installer/build-windows-installer.ps1 -Version "v1.0.0"
```

Requirements:
- `cargo`
- `makensis` (NSIS)
- The build script runs `packaging/windows/generate-assets.ps1` to render branded bitmap assets from `packaging/windows/design-tokens.json`.

### 3) macOS package (+ optional dmg)

```bash
bash packaging/macos/build-pkg.sh v1.0.0
```

Requirements:
- macOS host
- Xcode command line tools (`pkgbuild`)
- optional: `hdiutil` for DMG

### 4) Linux tarball

```bash
bash packaging/linux/build-tarball.sh v1.0.0 x86_64-unknown-linux-gnu
```

### 5) Linux `.deb` (optional)

```bash
bash packaging/linux/build-deb.sh v1.0.0
```

Requirements:
- Debian/Ubuntu environment
- `dpkg-deb`

### 6) Assemble release folder + checksums

```powershell
pwsh ./scripts/installer/assemble-release.ps1 -Version "release-v1.0.0"
```

```bash
bash scripts/installer/assemble-release.sh release-v1.0.0
```

## Install / Verify / Uninstall

### Windows
- Install: run `VoltaSetup-<version>.exe`
- Verify (installer runs this):
  - `volta version`
  - optional: `volta doctor --strict`
- Uninstall: `Uninstall Volta.exe` or Settings -> Installed apps

### macOS
- Admin flow: install `.pkg`
- User flow:

```bash
bash packaging/macos/install-user.sh --binary ./target/release/volta --verify-doctor
```

- Uninstall:

```bash
bash packaging/macos/uninstall.sh --user
# or
sudo bash packaging/macos/uninstall.sh --system
```

### Linux
- Install from binary/tarball:

```bash
bash packaging/linux/install.sh --binary ./target/release/volta --verify-doctor
# or
bash packaging/linux/install.sh --archive ./dist/release/v1.0.0/linux/volta-v1.0.0-x86_64-unknown-linux-gnu.tar.gz
```

- System install:

```bash
sudo bash packaging/linux/install.sh --binary ./target/release/volta --system
```

- Uninstall:

```bash
bash packaging/linux/uninstall.sh --user
# or
sudo bash packaging/linux/uninstall.sh --system
```

## Release Artifact Layout

```text
dist/release/<version>/
  windows/
    VoltaSetup-<version>.exe
  macos/
    Volta-<version>.pkg
    Volta-<version>.dmg
    install-volta-user.sh
    uninstall-volta.sh
  linux/
    volta-<version>-<target>.tar.gz
    volta_<version>_amd64.deb
    install.sh
    uninstall.sh
  checksums.txt
  release-notes.md
```

## Failure Handling Notes

- Installer writes useful logs:
  - Windows: `%LOCALAPPDATA%\\Volta\\bin\\installer.log`
  - macOS (pkg): `/var/log/volta-installer.log`
  - Linux user: `~/.volta/install.log`
  - Linux/macOS system: `/var/log/volta-installer.log`
- PATH updates are append-safe and avoid duplicates in user flows.
- Per-user install is default for scripts to avoid admin rights.

## V2 Improvements

- Code signing + notarization pipeline (Windows EV, macOS notarization)
- Rich Windows custom page assets exported directly from Figma
- RPM build lane and repo metadata
- GUI launcher for macOS/Linux (optional)
