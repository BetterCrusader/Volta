# RFC-2026-02-20-CUDA-INFERENCE-MVP

## Status

Accepted

## Context

Volta `v0.3` introduces CUDA inference support while preserving the single-truth runtime architecture:

- `ModelBuilder -> Verified IR -> ExecutionPlan -> Runtime`
- backend-neutral planner and execution-plan contracts
- no CUDA-specific control flow in train/infer gateway paths

## Decision

Adopt a phased CUDA inference MVP with the following contract constraints:

1. Runtime gateway remains the only execution entrypoint.
2. ExecutionPlan remains backend-neutral and receives placement hints without CUDA enums.
3. CUDA lowering maps kernel groups and placement hints to backend executable nodes without mutating the plan.
4. Unsupported CUDA kernels fail fast; no silent CPU fallback is allowed.
5. Strict determinism mode enforces fixed reduction topology and disables atomics/TF32/fast-math.
6. CPU/CUDA parity and CUDA memory regression are enforced by committed baselines in CI gates.

## Verification Gates

- `tests/cuda_backend_scaffold.rs`
- `tests/cuda_kernel_dispatch.rs`
- `tests/cuda_infer_determinism.rs`
- `tests/cuda_infer_parity.rs`
- `tests/cuda_infer_memory_guard.rs`
- `scripts/ci/cuda_infer_verify.sh`

## Risks

- MVP behavior is scaffolded and not yet a full CUDA kernel library.
- Determinism policy must be re-validated for every kernel-class expansion.

## Follow-ups

- Expand kernel coverage under the same no-fallback and strict-determinism rules.
- Maintain parity and memory baselines for each backend capability increase.
