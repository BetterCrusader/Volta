# CUDA Determinism Policy

## Scope

This policy defines deterministic execution guarantees for CUDA inference in Volta Phase 1.
All CUDA execution must flow through the same verified runtime gateway used by CPU.

## Determinism Levels

- `strict`
  - no atomics in kernel implementations
  - fixed reduction topology for softmax and other reductions
  - TF32 disabled
  - fast-math disabled
  - fail-fast if a kernel cannot satisfy strict requirements
- `balanced`
  - deterministic behavior is preferred but relaxed for compatibility
- `fast`
  - throughput-first mode with relaxed determinism guarantees

## Required Runtime Behavior

- Backend capability checks must reject strict mode on unsupported backends.
- CUDA compile/lowering must reject unsupported kernel classes; no silent CPU fallback is allowed.
- Strict mode must remain replayable for fixed inputs and fixed model parameters.

## Verification Gates

- `tests/cuda_infer_determinism.rs` validates strict replay and strict fail-fast behavior.
- `tests/cuda_infer_parity.rs` validates CPU/CUDA parity through the runtime gateway.
- `tests/cuda_infer_memory_guard.rs` validates deterministic memory contracts and placement mapping.

## Operational Notes

- Use `VOLTA_DETERMINISM=strict` in release validation paths.
- Any change affecting strict mode requires updating tests and baseline evidence in the same PR.
