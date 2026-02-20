#!/usr/bin/env bash
set -euo pipefail

echo "[cuda-train-verify] backward lowering gate"
cargo test --test cuda_train_backward -- --nocapture

echo "[cuda-train-verify] train e2e gate"
cargo test --test cuda_train_e2e -- --nocapture

echo "[cuda-train-verify] strict replay gate"
cargo test --test cuda_train_replay -- --nocapture

echo "[cuda-train-verify] training determinism artifact gate"
cargo test --test cuda_train_artifacts -- --nocapture

echo "[cuda-train-verify] optimizer parity gate"
cargo test --test cuda_optimizer_parity -- --nocapture

echo "[cuda-train-verify] memory guard gate"
cargo test --test cuda_train_memory_guard -- --nocapture

echo "[cuda-train-verify] runtime single-path regression"
cargo test --test runtime_single_truth_path -- --nocapture

echo "[cuda-train-verify] compile gate"
cargo test --no-run

echo "[cuda-train-verify] all CUDA training gates passed"
