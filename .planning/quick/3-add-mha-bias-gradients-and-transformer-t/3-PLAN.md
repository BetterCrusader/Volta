---
plan: 3
title: "add MHA bias gradients and transformer training parity"
status: completed
date: 2026-03-07
---

# Plan: MHA Bias Gradients and Transformer Training Parity

## Goal
Close the missing MHA bias-gradient path in autograd/interpreter and prove the fix with transformer training parity against PyTorch.

## Tasks

### T1: Extend MHA backward contract
- Files: `src/engine/ir/op.rs`, `src/engine/ir/autograd.rs`, `src/engine/ir/interpreter.rs`, `src/engine/ir/shape_inference.rs`, `src/engine/ir/fingerprint.rs`, `src/engine/ir/printer.rs`
- Action: Add `db_q/db_k/db_v/db_o` outputs to `MultiHeadAttentionBackward`, wire them through autograd accumulation, interpreter execution, shape inference, and helper metadata.
- Verify: `build_reverse_graph` can request MHA bias gradients without missing entries.

### T2: Expand PyTorch parity coverage
- Files: `examples/pytorch_parity.py`, `tests/pytorch_parity.rs`
- Action: Add bias-gradient parity for MHA/MHA self-attention and add a real transformer multi-step SGD parity case through `train_graph`.
- Verify: Transformer training parity passes with tight tolerances and MHA bias grads match PyTorch.

### T3: Re-run quality gates
- Files: workspace
- Action: Run formatter, tests, and clippy.
- Verify: `cargo fmt --all --check`, `cargo test --workspace --quiet`, and `cargo clippy --workspace --all-targets --quiet` all pass.
