#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HOME}/.volta"
BIN_DIR="$PREFIX/bin"
BINARY_PATH=""
SKIP_PATH="0"
VERIFY_DOCTOR="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      BINARY_PATH="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      BIN_DIR="$PREFIX/bin"
      shift 2
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

if [[ -z "$BINARY_PATH" ]]; then
  BINARY_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../target/release/volta"
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

if [[ "$SKIP_PATH" == "0" ]]; then
  append_profile_snippet "$HOME/.zshrc"
  append_profile_snippet "$HOME/.bashrc"
  append_profile_snippet "$HOME/.bash_profile"
fi

LOG_FILE="$PREFIX/install.log"
mkdir -p "$PREFIX"

{
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Volta user install"
  echo "prefix=$PREFIX"
  "$BIN_DIR/volta" version
  if [[ "$VERIFY_DOCTOR" == "1" ]]; then
    "$BIN_DIR/volta" doctor --json || true
  fi
} >> "$LOG_FILE" 2>&1

echo "Volta installed at $BIN_DIR/volta"
echo "Open a new shell or run: export PATH=\"$BIN_DIR:\$PATH\""
