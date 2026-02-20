#!/usr/bin/env bash
set -euo pipefail

VERIFY_ONLY=0
TARGET_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "error: --target requires a value" >&2
        exit 2
      fi
      TARGET_TAG="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

git fetch --tags --force

if [[ -z "$TARGET_TAG" ]]; then
  TARGET_TAG="$(git tag --sort=-creatordate | sed -n '2p')"
fi

if [[ -z "$TARGET_TAG" ]]; then
  if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    echo "rollback_target=none"
    echo "rollback verification skipped (no release tags found)"
    exit 0
  fi
  echo "error: unable to determine rollback target tag" >&2
  exit 1
fi

if ! git rev-parse --verify --quiet "refs/tags/$TARGET_TAG" >/dev/null; then
  echo "error: rollback target tag '$TARGET_TAG' does not exist" >&2
  exit 1
fi

echo "rollback_target=$TARGET_TAG"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  echo "rollback verification succeeded"
  exit 0
fi

echo "checking out rollback target '$TARGET_TAG'"
git checkout "$TARGET_TAG"
echo "rollback completed"
