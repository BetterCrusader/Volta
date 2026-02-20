#!/usr/bin/env bash
set -euo pipefail

echo "[fuzz-heavy] running large SSA fuzz corpus"
cargo test --release ir::freeze_hardening::tests::fuzz_ssa_graphs_5000_heavy -- --ignored --exact --nocapture

echo "[fuzz-heavy] running long determinism repeat"
cargo test --release ir::freeze_hardening::tests::long_run_determinism_100x50_heavy -- --ignored --exact --nocapture

echo "[fuzz-heavy] completed"
