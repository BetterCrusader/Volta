#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v1.0.0}"
TARGET="${2:-x86_64-unknown-linux-gnu}"
OUT_DIR="${3:-dist/release/${VERSION}/linux}"
BINARY_PATH="${4:-target/release/volta}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run on Linux" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "[linux-artifacts] release binary missing, building..."
  cargo build --release
fi

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "binary not found at $BINARY_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
cp "$BINARY_PATH" "$TMP_DIR/volta"
chmod 755 "$TMP_DIR/volta"

TARBALL="$OUT_DIR/volta-${VERSION}-${TARGET}.tar.gz"
tar -C "$TMP_DIR" -czf "$TARBALL" volta
rm -rf "$TMP_DIR"

sha256sum "$TARBALL" > "$TARBALL.sha256"
echo "[linux-artifacts] created $TARBALL"
