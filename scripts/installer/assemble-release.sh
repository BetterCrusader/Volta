#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-release-v1.0.0}"
ROOT="${2:-dist/release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RELEASE_ROOT="$ROOT/$VERSION"
WINDOWS_DIR="$RELEASE_ROOT/windows"
MACOS_DIR="$RELEASE_ROOT/macos"
LINUX_DIR="$RELEASE_ROOT/linux"

mkdir -p "$WINDOWS_DIR" "$MACOS_DIR" "$LINUX_DIR"

cp packaging/linux/install.sh "$LINUX_DIR/install.sh"
cp packaging/linux/uninstall.sh "$LINUX_DIR/uninstall.sh"
cp packaging/macos/install-user.sh "$MACOS_DIR/install-volta-user.sh"
cp packaging/macos/uninstall.sh "$MACOS_DIR/uninstall-volta.sh"

RELEASE_NOTES="$RELEASE_ROOT/release-notes.md"
if [[ ! -f "$RELEASE_NOTES" ]]; then
  cat > "$RELEASE_NOTES" <<EOF
# Volta $VERSION

## Highlights
- Deterministic CLI runtime and governance-gated release process.
- Cross-platform installer assets for Windows, macOS, and Linux.
- Verification-first install flow: volta version and volta doctor.

## Artifacts
- Windows: windows/VoltaSetup-<version>.exe
- macOS: macos/Volta-<version>.pkg and macos/Volta-<version>.dmg
- Linux: linux/volta-<version>-<target>.tar.gz and optional .deb
EOF
fi

CHECKSUM_FILE="$RELEASE_ROOT/checksums.txt"
: > "$CHECKSUM_FILE"

hash_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  else
    shasum -a 256 "$file" | awk '{print $1}'
  fi
}

while IFS= read -r -d '' file; do
  rel="${file#$RELEASE_ROOT/}"
  if [[ "$rel" == "checksums.txt" ]]; then
    continue
  fi
  hash="$(hash_file "$file")"
  printf '%s  %s\n' "$hash" "$rel" >> "$CHECKSUM_FILE"
done < <(find "$RELEASE_ROOT" -type f -print0 | sort -z)

echo "[assemble-release] release layout prepared at $RELEASE_ROOT"
