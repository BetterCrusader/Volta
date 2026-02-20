#!/usr/bin/env bash
set -euo pipefail

echo "[tiny-transformer-verify] acceptance parity gate"
cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture

echo "[tiny-transformer-verify] deterministic replay gate"
cargo test tiny_transformer_training_is_repeatable_for_fixed_seed -- --nocapture

echo "[tiny-transformer-verify] memory planner guard"
cargo test tiny_transformer_peak_memory_stays_within_budget -- --nocapture

echo "[tiny-transformer-verify] all milestone gates passed"
