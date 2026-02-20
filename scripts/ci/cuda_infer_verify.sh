#!/usr/bin/env bash
set -euo pipefail

echo "[cuda-infer-verify] backend scaffold gate"
cargo test --test cuda_backend_scaffold -- --nocapture

echo "[cuda-infer-verify] kernel dispatch gate"
cargo test --test cuda_kernel_dispatch -- --nocapture

echo "[cuda-infer-verify] strict determinism gate"
cargo test --test cuda_infer_determinism -- --nocapture

echo "[cuda-infer-verify] cpu-cuda parity gate"
cargo test --test cuda_infer_parity -- --nocapture

echo "[cuda-infer-verify] memory regression gate"
cargo test --test cuda_infer_memory_guard -- --nocapture

echo "[cuda-infer-verify] runtime single-path regression"
cargo test --test runtime_single_truth_path -- --nocapture

echo "[cuda-infer-verify] compile gate"
cargo test --no-run

echo "[cuda-infer-verify] all CUDA inference gates passed"
