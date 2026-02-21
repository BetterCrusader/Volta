#!/usr/bin/env bash
set -euo pipefail

THRESHOLD="${THRESHOLD_PERCENT:-5}"
BASELINE_DIR="${BASELINE_DIR:-benchmarks/baselines}"
REPORT_DIR="${REPORT_DIR:-benchmarks/reports}"

SIGNATURE="$(python scripts/perf/cpu_signature.py)"

mkdir -p "$REPORT_DIR"

run_single_pass() {
  local pass_id="$1"
  local probe_file="${REPORT_DIR}/release-perf-pass-${pass_id}-probe.json"
  local gate_file="${REPORT_DIR}/release-perf-pass-${pass_id}-gate.json"

  cargo run --release --bin perf_probe -- --samples 9 --dim 96 --matmul-iters 3 --relu-iters 16 > "$probe_file"

  if python scripts/perf/perf_gate.py \
      --signature "$SIGNATURE" \
      --baseline-dir "$BASELINE_DIR" \
      --threshold-percent "$THRESHOLD" \
      --probe-file "$probe_file" \
      --output-json "$gate_file"; then
    echo "pass_${pass_id}=pass"
    return 0
  fi

  echo "pass_${pass_id}=fail"
  return 1
}

PASS1_OK=0
PASS2_OK=0

if run_single_pass 1; then
  PASS1_OK=1
fi

if run_single_pass 2; then
  PASS2_OK=1
fi

resolve_baseline_file() {
  local gate_file="$1"
  python - "$gate_file" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("baseline_file", ""))
PY
}

BASELINE_FILE_PASS1="$(resolve_baseline_file "${REPORT_DIR}/release-perf-pass-1-gate.json")"
BASELINE_FILE_PASS2="$(resolve_baseline_file "${REPORT_DIR}/release-perf-pass-2-gate.json")"

if [[ -n "$BASELINE_FILE_PASS1" && -f "$BASELINE_FILE_PASS1" ]]; then
  python scripts/perf/baseline_compare.py \
    --baseline-file "$BASELINE_FILE_PASS1" \
    --candidate-file "${REPORT_DIR}/release-perf-pass-1-probe.json" \
    --output-markdown "${REPORT_DIR}/release-perf-pass-1.md" >/dev/null
fi

if [[ -n "$BASELINE_FILE_PASS2" && -f "$BASELINE_FILE_PASS2" ]]; then
  python scripts/perf/baseline_compare.py \
    --baseline-file "$BASELINE_FILE_PASS2" \
    --candidate-file "${REPORT_DIR}/release-perf-pass-2-probe.json" \
    --output-markdown "${REPORT_DIR}/release-perf-pass-2.md" >/dev/null
fi

echo "signature=${SIGNATURE}"
echo "threshold_percent=${THRESHOLD}"
echo "pass1_ok=${PASS1_OK}"
echo "pass2_ok=${PASS2_OK}"

if [[ "$PASS1_OK" -eq 0 && "$PASS2_OK" -eq 0 ]]; then
  echo "release perf gate failed on both passes" >&2
  exit 1
fi

echo "release perf gate passed"
