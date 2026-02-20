#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="${REPORT_DIR:-benchmarks/reports/nightly}"
mkdir -p "$REPORT_DIR"

for dim in 64 96 128 192; do
  echo "[nightly-perf] dim=${dim}"
  cargo run --release --bin perf_probe -- --samples 9 --dim "$dim" --matmul-iters 4 --relu-iters 16 > "${REPORT_DIR}/perf-dim-${dim}.json"
done

python scripts/perf/cpu_signature.py --json > "${REPORT_DIR}/cpu-signature.json"

echo "[nightly-perf] completed"
