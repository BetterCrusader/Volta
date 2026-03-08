---
plan: 2
title: "add multi-step PyTorch training parity through real train_graph"
status: completed
date: 2026-03-07
---

# Plan: Multi-Step Training Parity

## Goal
Prove that Volta's real `train_graph` SGD loop stays numerically aligned with PyTorch over multiple samples and epochs, not just a single manual optimizer step.

## Tasks

### T1: Extend the PyTorch oracle
- File: `examples/pytorch_parity.py`
- Action: Add a multi-step SGD training-loop case over a fixed two-sample dataset and emit final loss plus final parameters.
- Verify: Script returns deterministic JSON for `mlp_train_loop_sgd`.

### T2: Add Rust parity coverage through `train_graph`
- File: `tests/pytorch_parity.rs`
- Action: Build the same tiny MLP and dataset, run `train_graph` for multiple epochs, and compare final loss and final parameters against the PyTorch oracle.
- Verify: New parity test passes without relaxing tolerances to hide drift.

### T3: Re-run the quality gates
- Files: workspace
- Action: Run `cargo fmt --all --check`, `cargo test --workspace --quiet`, and `cargo clippy --workspace --all-targets --quiet`.
- Verify: All three gates stay green after the new parity coverage.
