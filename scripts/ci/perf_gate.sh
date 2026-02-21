#!/usr/bin/env bash
set -euo pipefail

THRESHOLD="${THRESHOLD_PERCENT:-5}"
BASELINE_DIR="${BASELINE_DIR:-benchmarks/baselines}"
REPORT_DIR="${REPORT_DIR:-benchmarks/reports/pr}"
ALLOW_MISSING_BASELINE="${ALLOW_MISSING_BASELINE:-0}"

mkdir -p "$REPORT_DIR"

PROBE_FILE="${REPORT_DIR}/pr-perf-probe.json"
GATE_FILE="${REPORT_DIR}/pr-perf-gate.json"
MARKDOWN_FILE="${REPORT_DIR}/pr-perf-summary.md"

echo "[perf-gate] running perf probe"
cargo run --release --bin perf_probe -- --samples 9 --dim 96 --matmul-iters 3 --relu-iters 16 >"$PROBE_FILE"

ALLOW_ARGS=()
if [[ "$ALLOW_MISSING_BASELINE" == "1" ]]; then
  ALLOW_ARGS+=(--allow-missing-baseline)
fi

echo "[perf-gate] evaluating threshold=${THRESHOLD}%"
python scripts/perf/perf_gate.py \
  --baseline-dir "$BASELINE_DIR" \
  --threshold-percent "$THRESHOLD" \
  --probe-file "$PROBE_FILE" \
  --output-json "$GATE_FILE" \
  "${ALLOW_ARGS[@]}"

BASELINE_FILE="$(python - "$GATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("baseline_file", ""))
PY
)"

if [[ -n "$BASELINE_FILE" && -f "$BASELINE_FILE" ]]; then
  python scripts/perf/baseline_compare.py \
    --baseline-file "$BASELINE_FILE" \
    --candidate-file "$PROBE_FILE" \
    --output-markdown "$MARKDOWN_FILE" >/dev/null
fi

echo "[perf-gate] gate report: ${GATE_FILE}"
echo "[perf-gate] completed"
