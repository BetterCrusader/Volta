#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HOME}/.volta"
BIN_DIR="$PREFIX/bin"
ARCHIVE_PATH=""
BINARY_PATH=""
SYSTEM_MODE="0"
SKIP_PATH="0"
VERIFY_DOCTOR="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      ARCHIVE_PATH="$2"
      shift 2
      ;;
    --binary)
      BINARY_PATH="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      BIN_DIR="$PREFIX/bin"
      shift 2
      ;;
    --system)
      SYSTEM_MODE="1"
      PREFIX="/usr/local"
      BIN_DIR="$PREFIX/bin"
      shift
      ;;
    --skip-path)
      SKIP_PATH="1"
      shift
      ;;
    --verify-doctor)
      VERIFY_DOCTOR="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$SYSTEM_MODE" == "1" ]] && [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "--system install requires sudo/root" >&2
  exit 1
fi

TMP_DIR=""
cleanup() {
  if [[ -n "$TMP_DIR" ]] && [[ -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

if [[ -n "$ARCHIVE_PATH" ]]; then
  if [[ ! -f "$ARCHIVE_PATH" ]]; then
    echo "archive not found: $ARCHIVE_PATH" >&2
    exit 1
  fi
  TMP_DIR="$(mktemp -d)"
  tar -C "$TMP_DIR" -xzf "$ARCHIVE_PATH"
  BINARY_PATH="$TMP_DIR/volta"
fi

if [[ -z "$BINARY_PATH" ]]; then
  CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../target/release/volta"
  if [[ -f "$CANDIDATE" ]]; then
    BINARY_PATH="$CANDIDATE"
  else
    echo "binary not provided. Use --binary <path> or --archive <tar.gz>" >&2
    exit 1
  fi
fi

if [[ ! -f "$BINARY_PATH" ]]; then
  echo "binary not found at $BINARY_PATH" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"
install -m 755 "$BINARY_PATH" "$BIN_DIR/volta"

SNIPPET_START="# >>> volta installer >>>"
SNIPPET_END="# <<< volta installer <<<"
SNIPPET_BODY="export PATH=\"$BIN_DIR:\$PATH\""

append_profile_snippet() {
  local rc="$1"
  touch "$rc"
  if ! grep -Fq "$SNIPPET_START" "$rc"; then
    {
      echo ""
      echo "$SNIPPET_START"
      echo "$SNIPPET_BODY"
      echo "$SNIPPET_END"
    } >> "$rc"
  fi
}

if [[ "$SKIP_PATH" == "0" ]] && [[ "$SYSTEM_MODE" == "0" ]]; then
  append_profile_snippet "$HOME/.bashrc"
  append_profile_snippet "$HOME/.zshrc"
fi

LOG_FILE="$PREFIX/install.log"
if [[ "$SYSTEM_MODE" == "1" ]]; then
  LOG_FILE="/var/log/volta-installer.log"
fi

{
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Linux install"
  echo "prefix=$PREFIX"
  "$BIN_DIR/volta" version
  if [[ "$VERIFY_DOCTOR" == "1" ]]; then
    "$BIN_DIR/volta" doctor --json || true
  fi
} >> "$LOG_FILE" 2>&1

if [[ "$SYSTEM_MODE" == "1" ]]; then
  echo "Volta installed at $BIN_DIR/volta"
else
  echo "Volta installed at $BIN_DIR/volta"
  echo "Open a new shell or run: export PATH=\"$BIN_DIR:\$PATH\""
fi
