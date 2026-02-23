#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v py >/dev/null 2>&1; then
  PYTHON_BIN="py"
else
  echo "[wave1-verify] python interpreter not found (tried: python, python3, py)"
  exit 1
fi

echo "[wave1-verify] cargo fmt --check"
cargo fmt --check

echo "[wave1-verify] cargo clippy --all-targets --all-features -- -D warnings"
cargo clippy --all-targets --all-features -- -D warnings

echo "[wave1-verify] cargo test"
cargo test

echo "[wave1-verify] cargo test --release"
cargo test --release

echo "[wave1-verify] cargo test --test property_fast"
cargo test --test property_fast

echo "[wave1-verify] ${PYTHON_BIN} scripts/ci/detect_tiers.py --paths src/ir/tensor.rs"
tier_output="$(${PYTHON_BIN} scripts/ci/detect_tiers.py --paths src/ir/tensor.rs)"
printf '%s\n' "$tier_output"
if [[ "$tier_output" != *"tier=A"* ]]; then
  echo "[wave1-verify] detect_tiers sanity check failed"
  exit 1
fi

echo "[wave1-verify] ${PYTHON_BIN} scripts/ci/policy_check.py --paths src/ir/tensor.rs --pr-body 'RFC-004'"
${PYTHON_BIN} scripts/ci/policy_check.py --paths src/ir/tensor.rs --pr-body "RFC-004"

echo "[wave1-verify] bash scripts/ci/cli_smoke.sh"
bash scripts/ci/cli_smoke.sh

echo "[wave1-verify] all checks passed"
