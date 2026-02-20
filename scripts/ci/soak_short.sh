#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:-3}"

echo "[soak-short] runs=${RUNS}"

for run in $(seq 1 "$RUNS"); do
  echo "[soak-short] iteration ${run}/${RUNS}: deterministic training repeat"
  cargo test --release ir::freeze_hardening::tests::deterministic_training_repeats_match -- --exact

  echo "[soak-short] iteration ${run}/${RUNS}: memory pressure stability"
  cargo test --release ir::freeze_hardening::tests::memory_pressure_plan_is_stable -- --exact
done

echo "[soak-short] completed"
