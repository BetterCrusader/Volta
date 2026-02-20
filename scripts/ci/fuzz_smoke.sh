#!/usr/bin/env bash
set -euo pipefail

echo "[fuzz-smoke] verifier-guarded IR fuzz"
cargo test --release ir::freeze_hardening::tests::fuzz_ssa_graphs_with_verifier_guards -- --exact

echo "[fuzz-smoke] chaos pass repeatability"
cargo test --release ir::freeze_hardening::tests::pass_order_chaos_mode_is_repeatable_per_seed -- --exact

echo "[fuzz-smoke] completed"
