# Quick Task 4: expand train_graph parity coverage - Plan

**Gathered:** 2026-03-07
**Status:** In Progress

## Task 1
- files: [examples/pytorch_parity.py](/C:/Users/User/Desktop/Volta-main/examples/pytorch_parity.py)
- action: extend the PyTorch oracle with broader train-loop coverage for `Adam`, `AdamW`, `gradient_accumulation_steps=2`, and `clip_grad`, while factoring shared MLP/transformer helpers instead of copying more loops
- verify: `python -m py_compile examples/pytorch_parity.py`
- done: oracle emits deterministic JSON for all new train-loop parity cases

## Task 2
- files: [tests/pytorch_parity.rs](/C:/Users/User/Desktop/Volta-main/tests/pytorch_parity.rs)
- action: add Rust integration tests that run the real `train_graph` with optimizer parity, accumulation, and clip-grad enabled and compare final loss and parameters against the PyTorch oracle
- verify: targeted `cargo test` for the new parity tests
- done: all new parity tests are green

## Task 3
- files: [tests/pytorch_parity.rs](/C:/Users/User/Desktop/Volta-main/tests/pytorch_parity.rs), [src/engine/ir/train.rs](/C:/Users/User/Desktop/Volta-main/src/engine/ir/train.rs), [src/engine/ir/optimizer.rs](/C:/Users/User/Desktop/Volta-main/src/engine/ir/optimizer.rs)
- action: fix any correctness mismatch exposed by the new parity cases, tighten weak optimizer validation if parity reveals a contract gap, then run full validation
- verify: `cargo fmt --all --check`, `cargo test --workspace --quiet`, `cargo clippy --workspace --all-targets --quiet`
- done: full workspace checks stay green after the new coverage
