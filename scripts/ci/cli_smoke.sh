#!/usr/bin/env bash
set -euo pipefail

run_step() {
  echo "[cli-smoke] $*"
  local log_file
  log_file=$(mktemp)
  if ! "$@" >"$log_file" 2>&1; then
    cat "$log_file"
    rm -f "$log_file"
    return 1
  fi
  rm -f "$log_file"
}

run_step cargo run -- --help
run_step cargo run -- examples/mnist.vt
run_step cargo run -- --bench-infer --runs 1 --warmup 0 --tokens 4
run_step cargo run -- --tune-matmul --dim 64 --runs 1

echo "[cli-smoke] all checks passed"
