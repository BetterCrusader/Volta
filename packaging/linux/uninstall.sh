#!/usr/bin/env bash
set -euo pipefail

MODE="user"
PREFIX="${HOME}/.volta"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      MODE="system"
      shift
      ;;
    --user)
      MODE="user"
      shift
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

remove_profile_snippet() {
  local rc="$1"
  local tmp
  [[ -f "$rc" ]] || return 0
  tmp="$(mktemp)"
  awk '
    BEGIN { skip = 0 }
    /^# >>> volta installer >>>$/ { skip = 1; next }
    /^# <<< volta installer <<<$/ { skip = 0; next }
    { if (!skip) print $0 }
  ' "$rc" > "$tmp"
  mv "$tmp" "$rc"
}

if [[ "$MODE" == "system" ]]; then
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "system uninstall requires sudo/root" >&2
    exit 1
  fi

  rm -f /usr/local/bin/volta
  rm -f /var/log/volta-installer.log
  echo "Removed system-wide Volta installation"
  exit 0
fi

rm -f "$PREFIX/bin/volta"
rm -f "$PREFIX/install.log"
rmdir "$PREFIX/bin" 2>/dev/null || true
rmdir "$PREFIX" 2>/dev/null || true

remove_profile_snippet "$HOME/.bashrc"
remove_profile_snippet "$HOME/.zshrc"

echo "Removed user Volta installation"
