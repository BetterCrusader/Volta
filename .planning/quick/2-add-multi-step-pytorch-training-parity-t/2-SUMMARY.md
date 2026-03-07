---
plan: 2
status: completed
date: 2026-03-07
---

# Summary: Multi-Step Training Parity

## What was done
- Wired in the broader PyTorch parity harness for MLP, Conv2D, LayerNorm, BatchNorm, MHA, self-attention, transformer block, and SGD training cases.
- Fixed the MHA backward contract so recomputed projections carry the correct bias tensors, which unblocked self-attention and transformer parity.
- Fixed the transformer builder so the encoder block emits a valid MHA family for autograd and uses a supported flattening path.
- Extended the PyTorch oracle with a real multi-step SGD loop over a fixed two-sample dataset.
- Added a Rust integration test that drives the same tiny MLP through `train_graph` for three epochs and compares final loss plus final parameters against PyTorch.
- Re-ran the workspace quality gates after the new parity coverage landed.

## Result
- PyTorch is now an external numerical oracle for the main compiled-training path instead of only internal Volta-vs-Volta checks.
- Volta now has parity coverage for a real multi-step training loop, not just one manual optimizer step.
- The new test proves `train_graph` stays aligned with PyTorch across repeated updates on the same model and dataset.

## Files changed
- `examples/pytorch_parity.py`
- `tests/pytorch_parity.rs`
- `src/engine/ir/autograd.rs`
- `src/engine/ir/fingerprint.rs`
- `src/engine/ir/interpreter.rs`
- `src/engine/ir/op.rs`
- `src/engine/ir/printer.rs`
- `src/engine/ir/transformer.rs`

## Verification
- `cargo fmt --all --check`
- `cargo test --workspace --quiet`
- `cargo clippy --workspace --all-targets --quiet`
