# Quick Task 4: expand train_graph parity coverage - Summary

**Completed:** 2026-03-07
**Code Commit:** `148e9c7`
**Status:** Done

## What changed

- Extended the PyTorch oracle in [examples/pytorch_parity.py](/C:/Users/User/Desktop/Volta-main/examples/pytorch_parity.py) with train-loop parity cases for:
  - `MLP + Adam`
  - `MLP + AdamW`
  - `MLP + SGD + gradient_accumulation_steps=2`
  - `MLP + SGD + clip_grad=0.1`
  - `Transformer block + Adam`
  - `Transformer block + AdamW`
- Refactored the parity oracle to reuse shared MLP and transformer training helpers instead of duplicating each loop.
- Extended [tests/pytorch_parity.rs](/C:/Users/User/Desktop/Volta-main/tests/pytorch_parity.rs) so the real `train_graph` path is now checked against PyTorch for all of the above.
- Tightened [src/engine/ir/optimizer.rs](/C:/Users/User/Desktop/Volta-main/src/engine/ir/optimizer.rs) by adding missing `AdamW` hyperparameter validation and a regression test for invalid `beta1`, `epsilon`, and `weight_decay`.

## Result

- No new numeric mismatch was found.
- `train_graph` now has parity coverage not just for plain SGD, but also for:
  - stateful optimizers (`Adam`, `AdamW`)
  - gradient accumulation
  - gradient clipping

## Validation

- `python -m py_compile examples/pytorch_parity.py`
- `cargo test pytorch_parity_mlp_multi_step_sgd_accum2_train_graph -- --nocapture`
- `cargo test pytorch_parity_mlp_multi_step_sgd_clip_grad_train_graph -- --nocapture`
- `cargo test pytorch_parity_mlp_multi_step_adam_train_graph -- --nocapture`
- `cargo test pytorch_parity_mlp_multi_step_adamw_train_graph -- --nocapture`
- `cargo test pytorch_parity_transformer_multi_step_adam_train_graph -- --nocapture`
- `cargo test pytorch_parity_transformer_multi_step_adamw_train_graph -- --nocapture`
- `cargo fmt --all --check`
- `cargo test --workspace --quiet`
- `cargo clippy --workspace --all-targets --quiet`

## Honest takeaway

This task raised the correctness bar, but it still does not prove the whole training stack is safe. The uncovered surface is smaller now, not gone. The next high-value checks are `RmsProp/Adagrad` parity or applying accumulation/clipping parity on the transformer path instead of only MLP.
