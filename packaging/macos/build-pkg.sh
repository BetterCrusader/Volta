#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v1.0.0}"
OUT_DIR="${2:-dist/release/${VERSION}/macos}"
BINARY_PATH="${3:-target/release/volta}"
BUILD_DMG="${BUILD_DMG:-1}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must run on macOS" >&2
  exit 1
fi

if ! command -v pkgbuild >/dev/null 2>&1; then
  echo "pkgbuild is required (Xcode command line tools)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "[macos-installer] release binary missing, building..."
  cargo build --release
fi

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "binary not found at $BINARY_PATH" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
ROOT_DIR="$TMP_DIR/pkg-root"
SCRIPTS_DIR="$TMP_DIR/scripts"
mkdir -p "$ROOT_DIR/usr/local/lib/volta/bin" "$SCRIPTS_DIR" "$OUT_DIR"

cp "$BINARY_PATH" "$ROOT_DIR/usr/local/lib/volta/bin/volta"
chmod 755 "$ROOT_DIR/usr/local/lib/volta/bin/volta"
cp packaging/macos/scripts/preinstall "$SCRIPTS_DIR/preinstall"
cp packaging/macos/scripts/postinstall "$SCRIPTS_DIR/postinstall"
chmod 755 "$SCRIPTS_DIR/preinstall" "$SCRIPTS_DIR/postinstall"

PKG_PATH="$OUT_DIR/Volta-${VERSION}.pkg"
pkgbuild \
  --root "$ROOT_DIR" \
  --scripts "$SCRIPTS_DIR" \
  --identifier "dev.volta.cli" \
  --version "$VERSION" \
  --install-location "/" \
  "$PKG_PATH"

cp packaging/macos/uninstall.sh "$OUT_DIR/uninstall-volta.sh"
chmod 755 "$OUT_DIR/uninstall-volta.sh"
cp packaging/macos/install-user.sh "$OUT_DIR/install-volta-user.sh"
chmod 755 "$OUT_DIR/install-volta-user.sh"

if [[ "$BUILD_DMG" == "1" ]] && command -v hdiutil >/dev/null 2>&1; then
  DMG_DIR="$TMP_DIR/dmg"
  mkdir -p "$DMG_DIR"
  cp "$PKG_PATH" "$DMG_DIR/"
  cp "$OUT_DIR/uninstall-volta.sh" "$DMG_DIR/"
  cp "$OUT_DIR/install-volta-user.sh" "$DMG_DIR/"
  hdiutil create \
    -volname "Volta Installer" \
    -srcfolder "$DMG_DIR" \
    -ov \
    -format UDZO \
    "$OUT_DIR/Volta-${VERSION}.dmg" >/dev/null
fi

rm -rf "$TMP_DIR"

echo "[macos-installer] artifacts in $OUT_DIR"
