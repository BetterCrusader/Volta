#!/usr/bin/env sh
# Volta installer for Linux and macOS.
# Detects OS and architecture, downloads the correct binary, installs to ~/.local/bin.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/BetterCrusader/Volta/main/install/install.sh | sh
#
# Options (environment variables):
#   VOLTA_VERSION  — version to install (default: latest)
#   VOLTA_PREFIX   — install directory (default: ~/.local/bin)

set -e

REPO="BetterCrusader/Volta"
BINARY="volta"
PREFIX="${VOLTA_PREFIX:-$HOME/.local/bin}"

# ── Detect OS ──────────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Linux)  OS="linux" ;;
    Darwin) OS="macos" ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please build from source: https://github.com/$REPO"
        exit 1
        ;;
esac

# ── Detect architecture ────────────────────────────────────────────────────────
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64 | amd64) ARCH="x86_64" ;;
    aarch64 | arm64) ARCH="aarch64" ;;
    *)
        echo "Unsupported architecture: $ARCH"
        echo "Please build from source: https://github.com/$REPO"
        exit 1
        ;;
esac

# ── Resolve version ────────────────────────────────────────────────────────────
if [ -z "$VOLTA_VERSION" ]; then
    if command -v curl > /dev/null 2>&1; then
        VOLTA_VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
            | grep '"tag_name"' | sed 's/.*"tag_name": "\(.*\)".*/\1/')"
    elif command -v wget > /dev/null 2>&1; then
        VOLTA_VERSION="$(wget -qO- "https://api.github.com/repos/$REPO/releases/latest" \
            | grep '"tag_name"' | sed 's/.*"tag_name": "\(.*\)".*/\1/')"
    else
        echo "curl or wget is required."
        exit 1
    fi
fi

if [ -z "$VOLTA_VERSION" ]; then
    echo "Could not determine latest version. Set VOLTA_VERSION manually."
    exit 1
fi

echo "Installing Volta $VOLTA_VERSION ($OS/$ARCH)..."

# ── Build download URL ─────────────────────────────────────────────────────────
# Release asset naming: volta-<version>-<arch>-<os>.tar.gz
# Example: volta-v1.2.0-x86_64-linux.tar.gz
ASSET="volta-${VOLTA_VERSION}-${ARCH}-${OS}.tar.gz"
URL="https://github.com/$REPO/releases/download/$VOLTA_VERSION/$ASSET"

# ── Download ───────────────────────────────────────────────────────────────────
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL..."
if command -v curl > /dev/null 2>&1; then
    curl -fsSL "$URL" -o "$TMP/$ASSET"
elif command -v wget > /dev/null 2>&1; then
    wget -qO "$TMP/$ASSET" "$URL"
fi

# ── Extract ────────────────────────────────────────────────────────────────────
tar -xzf "$TMP/$ASSET" -C "$TMP"

# ── Install ────────────────────────────────────────────────────────────────────
mkdir -p "$PREFIX"
install -m 755 "$TMP/$BINARY" "$PREFIX/$BINARY"

echo ""
echo "Volta installed to $PREFIX/volta"

# ── PATH hint ─────────────────────────────────────────────────────────────────
case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *)
        echo ""
        echo "Add to your shell profile to use 'volta' anywhere:"
        echo "  export PATH=\"$PREFIX:\$PATH\""
        ;;
esac

echo ""
echo "Run: volta --version"
