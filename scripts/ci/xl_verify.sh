#!/usr/bin/env bash
set -euo pipefail

echo "[xl-verify] execution plan cache gate"
cargo test --test execution_plan_cache -- --nocapture

echo "[xl-verify] static memory budget gate"
cargo test --test xl_static_memory_budget -- --nocapture

echo "[xl-verify] gradient checkpointing gate"
cargo test --test xl_gradient_checkpointing -- --nocapture

echo "[xl-verify] backend capability matrix gate"
cargo test --test backend_capability_matrix -- --nocapture

echo "[xl-verify] pass equivalence gate"
cargo test --test pass_equivalence -- --nocapture

echo "[xl-verify] model export gate"
cargo test --test model_export -- --nocapture

echo "[xl-verify] runtime single-path regression"
cargo test --test runtime_single_truth_path -- --nocapture

echo "[xl-verify] compile gate"
cargo test --no-run

echo "[xl-verify] all phase 3 gates passed"
