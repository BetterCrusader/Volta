#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v1.0.0}"
OUT_DIR="${2:-dist/release/${VERSION}/linux}"
BINARY_PATH="${3:-target/release/volta}"
ARCH="amd64"
DEB_VERSION="${VERSION#release-}"
DEB_VERSION="${DEB_VERSION#v}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run on Linux" >&2
  exit 1
fi

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required for .deb build" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "[linux-deb] release binary missing, building..."
  cargo build --release
fi

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "binary not found at $BINARY_PATH" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
mkdir -p "$STAGE/DEBIAN" "$STAGE/usr/bin"
cp "$BINARY_PATH" "$STAGE/usr/bin/volta"
chmod 755 "$STAGE/usr/bin/volta"

cat > "$STAGE/DEBIAN/control" <<EOF
Package: volta
Version: ${DEB_VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: Volta OSS <oss@volta.dev>
Description: Volta deterministic AI compiler CLI
 A compiler-first ML runtime focused on deterministic execution.
EOF

cp packaging/linux/deb/postinst "$STAGE/DEBIAN/postinst"
cp packaging/linux/deb/prerm "$STAGE/DEBIAN/prerm"
chmod 755 "$STAGE/DEBIAN/postinst" "$STAGE/DEBIAN/prerm"

mkdir -p "$OUT_DIR"
DEB_PATH="$OUT_DIR/volta_${DEB_VERSION}_${ARCH}.deb"
dpkg-deb --build "$STAGE" "$DEB_PATH" >/dev/null
rm -rf "$STAGE"

echo "[linux-deb] created $DEB_PATH"
