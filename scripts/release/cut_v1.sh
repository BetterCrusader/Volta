#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-release-v1.0.0}"
REMOTE="${2:-origin}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[cut-v1] working tree is not clean"
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "[cut-v1] must run from main branch (current: $CURRENT_BRANCH)"
  exit 1
fi

echo "[cut-v1] syncing with $REMOTE/main"
git fetch "$REMOTE"
git merge --ff-only "$REMOTE/main"

echo "[cut-v1] running release checks"
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
bash scripts/ci/interop_onnx_verify.sh
bash scripts/ci/xl_verify.sh
bash scripts/release/rollback.sh --verify-only

echo "[cut-v1] creating tag: $TAG"
git tag -a "$TAG" -m "Volta V1 release"
git push "$REMOTE" "$TAG"
echo "[cut-v1] done"
