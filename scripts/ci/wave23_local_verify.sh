#!/usr/bin/env bash
set -euo pipefail

echo "[wave23-verify] running wave1 baseline verification"
bash scripts/ci/wave1_local_verify.sh

echo "[wave23-verify] python perf helper tests"
python -m unittest scripts.perf.tests.test_cpu_signature scripts.perf.tests.test_perf_gate scripts.perf.tests.test_baseline_compare -v

echo "[wave23-verify] validating perf gate wiring with deterministic probe input"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
PROBE_FILE="${TMP_DIR}/probe.json"

cargo run --release --bin perf_probe -- --samples 9 --dim 96 --matmul-iters 3 --relu-iters 16 > "$PROBE_FILE"
python scripts/perf/perf_gate.py --allow-missing-baseline --threshold-percent 5 --baseline-dir "$TMP_DIR" --signature "local-verify" --probe-file "$PROBE_FILE"
python scripts/perf/perf_gate.py --threshold-percent 5 --baseline-dir "$TMP_DIR" --signature "local-verify" --probe-file "$PROBE_FILE"

echo "[wave23-verify] fuzz smoke"
bash scripts/ci/fuzz_smoke.sh

echo "[wave23-verify] short soak"
bash scripts/ci/soak_short.sh 1

echo "[wave23-verify] rollback verify"
bash scripts/release/rollback.sh --verify-only

echo "[wave23-verify] all checks passed"
