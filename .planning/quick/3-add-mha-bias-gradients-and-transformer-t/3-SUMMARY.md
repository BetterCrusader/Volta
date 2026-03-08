---
plan: 3
status: completed
date: 2026-03-07
---

# Summary: MHA Bias Gradients and Transformer Training Parity

## What was done
- Extended `MultiHeadAttentionBackward` to produce all missing bias gradients: `db_q`, `db_k`, `db_v`, and `db_o`.
- Wired those new outputs through autograd accumulation, interpreter execution, shape inference, fingerprinting, and graph printing.
- Extended the PyTorch parity harness to check MHA bias gradients and added a multi-step transformer SGD parity case through the real `train_graph`.

## Result
- MHA bias parameters no longer fall out of the backward graph.
- Self-attention and transformer training parity now stay aligned with PyTorch after repeated optimizer updates, not just single-step forward/gradient checks.

## Files changed
- `src/engine/ir/op.rs`
- `src/engine/ir/autograd.rs`
- `src/engine/ir/interpreter.rs`
- `src/engine/ir/shape_inference.rs`
- `src/engine/ir/fingerprint.rs`
- `src/engine/ir/printer.rs`
- `examples/pytorch_parity.py`
- `tests/pytorch_parity.rs`

## Verification
- `cargo fmt --all --check`
- `cargo test --workspace --quiet`
- `cargo clippy --workspace --all-targets --quiet`
