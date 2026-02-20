#!/usr/bin/env bash
set -euo pipefail

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

echo "[wave1-verify] python -m unittest scripts.ci.tests.test_detect_tiers -v"
python -m unittest scripts.ci.tests.test_detect_tiers -v

echo "[wave1-verify] python -m unittest scripts.ci.tests.test_policy_check -v"
python -m unittest scripts.ci.tests.test_policy_check -v

echo "[wave1-verify] bash scripts/ci/cli_smoke.sh"
bash scripts/ci/cli_smoke.sh

echo "[wave1-verify] all checks passed"
