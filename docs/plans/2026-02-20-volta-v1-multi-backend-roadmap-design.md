# Volta v1 Multi-Backend Roadmap Design

## Status

Approved roadmap baseline for next 12 months.

## Goal

Evolve Volta from v0.2.0 CPU milestone into a production-grade, deterministic, multi-backend ML compiler-runtime without introducing a second execution truth path.

## Non-Negotiable Architecture Invariants

1. Single execution truth path:
   `ModelBuilder -> verify_graph -> build_execution_plan -> runtime gateway -> backend execution`
2. No backend-specific semantics in core IR.
3. Determinism is explicit policy, never implicit behavior.
4. Strict mode fails fast on unsupported deterministic behavior.
5. Every milestone ships with:
   - acceptance E2E test
   - determinism replay test
   - memory regression guard
   - CI verification script
   - design + implementation plan docs

## Target State

- CUDA inference and training support with deterministic mode.
- Backend capability abstraction for CPU/CUDA and future design-only backends.
- Runtime performance layer with plan cache and robust memory accounting.
- XL scaling support with static memory budgeting and checkpointed training support.
- Stable public DSL/API and model export format.

## Phase 1: CUDA Inference MVP (v0.3.x)

### Scope

- Inference-only CUDA backend.
- CPU/CUDA parity harness.
- Backend-neutral execution plan with backend placement hints.
- Strict deterministic CUDA mode.
- GPU-aware memory planning and guards.

### Exact Modules to Add

- `src/ir/runtime.rs`
- `src/ir/backend_capabilities.rs`
- `src/ir/cuda/mod.rs`
- `src/ir/cuda/device.rs`
- `src/ir/cuda/lowering.rs`
- `src/ir/cuda/executor.rs`
- `src/ir/cuda/memory.rs`
- `src/ir/cuda/determinism.rs`
- `src/ir/cuda/kernels/mod.rs`
- `src/ir/cuda/kernels/matmul.rs`
- `src/ir/cuda/kernels/add.rs`
- `src/ir/cuda/kernels/relu.rs`
- `src/ir/cuda/kernels/softmax.rs`

### Exact Files to Modify

- `src/ir/mod.rs`
- `src/ir/backend.rs`
- `src/ir/execution_plan.rs`
- `src/ir/allocation.rs`
- `src/ir/memory_planner.rs`
- `src/ir/shape_inference.rs`
- `src/model/builder.rs`
- `src/model/train_api.rs`
- `src/ir/train.rs`
- `src/ir/compiler_flags.rs`
- `.github/workflows/pr-gates.yml`
- `.github/workflows/nightly-quality.yml`
- `README.md`

### Verification Assets to Add

- `tests/cuda_infer_parity.rs`
- `tests/cuda_infer_determinism.rs`
- `tests/cuda_infer_memory_guard.rs`
- `tests/runtime_single_truth_path.rs`
- `scripts/ci/cuda_infer_verify.sh`
- `benchmarks/baselines/cuda-infer-parity.json`
- `docs/governance/cuda-determinism-policy.md`

### Determinism Risks

- Reduction ordering nondeterminism.
- Implicit mixed-precision behavior divergence.
- Silent CPU fallback masking CUDA issues.

### Memory Model Changes

- Shape-bound byte accounting must become non-zero and meaningful.
- Introduce backend memory classes (host input, device param, device temp, device output).
- Add transfer accounting (H2D/D2H edges).

## Phase 2: CUDA Training (v0.4.x)

### Scope

- Backward kernels and optimizer support on CUDA.
- Deterministic reduction and accumulation policies.
- Seed-stable CUDA training replay.
- Optional mixed precision, deterministic strict mode remains mandatory.

### Exact Modules to Add

- `src/ir/cuda/kernels/backward.rs`
- `src/ir/cuda/kernels/reductions.rs`
- `src/ir/cuda/train_executor.rs`

### Exact Files to Modify

- `src/ir/autograd.rs`
- `src/ir/train.rs`
- `src/model/train_api.rs`
- `src/ir/optimizer.rs`
- `src/ir/runtime.rs`
- `src/ir/backend.rs`
- `.github/workflows/release-gates.yml`
- `.github/workflows/nightly-quality.yml`

### Verification Assets to Add

- `tests/cuda_train_e2e.rs`
- `tests/cuda_train_replay.rs`
- `tests/cuda_train_memory_guard.rs`
- `scripts/ci/cuda_train_verify.sh`
- `benchmarks/baselines/cuda-train-replay.json`
- `docs/governance/cuda-training-determinism.md`

### Determinism Risks

- Atomic update nondeterminism in gradient accumulation.
- Optimizer state drift due update ordering.
- Replay instability when reduction tree is not fixed.

### Memory Model Changes

- Track gradient storage classes explicitly.
- Add deterministic accumulation workspace accounting.
- Expand planner for train-time temporary lifetimes.

## Phase 3: Performance + XL Scaling (v0.5.x to v1.0)

### Scope

- Execution plan caching in compiled models.
- Robust byte-level memory accounting.
- Schedule and fusion expansions.
- XL model support with memory-aware planning.

### Exact Modules to Add

- `src/ir/plan_cache.rs`
- `src/ir/schedule_optimization.rs`
- `src/ir/static_memory_budget.rs`
- `src/model/scaling.rs`
- `src/model/gradient_checkpointing.rs`
- `src/model/export.rs`

### Exact Files to Modify

- `src/model/builder.rs`
- `src/ir/execution_plan.rs`
- `src/ir/kernel_grouping.rs`
- `src/ir/memory_planner.rs`
- `src/ir/backend.rs`
- `src/main.rs`
- `src/lib.rs`
- `.github/workflows/release-gates.yml`
- `.github/workflows/nightly-quality.yml`

### Verification Assets to Add

- `tests/execution_plan_cache.rs`
- `tests/xl_static_memory_budget.rs`
- `tests/xl_gradient_checkpointing.rs`
- `tests/backend_capability_matrix.rs`
- `scripts/ci/xl_verify.sh`
- `docs/governance/backend-capability-matrix.md`
- `docs/governance/perf-slo.md`

### Determinism Risks

- Pass ordering instability under aggressive optimization.
- Cache key collisions causing stale executable reuse.
- XL fallback behavior diverging from strict determinism policy.

### Memory Model Changes

- Static peak budget reports by shape and batch profile.
- Recompute-aware planning for gradient checkpointing.
- Device/host budget partitions for large models.

## IR Invariants That Must Not Break

1. SSA single assignment remains enforced.
2. `verify_graph` executes before any lowering/scheduling/allocation.
3. `verify_schedule` and `verify_allocation` remain mandatory checks.
4. `ExecutionPlan` remains backend-neutral contract.
5. `Op` enum does not gain backend-only semantics in core pipeline.
6. Determinism mode is explicit runtime/compiler input, not hidden default.

## Risk Register Summary

1. **Second executor drift**: blocked by runtime-single-path tests.
2. **Determinism regressions**: blocked by replay tests + strict policy.
3. **Memory accounting blind spots**: blocked by non-zero byte guard and baselines.
4. **CI governance deadlocks**: required PR checks only for PR-triggered workflows.
5. **Backend mismatch bugs**: blocked by capability contracts and parity matrix.

## Release and Branching Strategy

- `milestone/cuda-inference-mvp-v0.3`
- `milestone/cuda-training-v0.4`
- `milestone/perf-xl-v0.5`

Each milestone follows:

1. design doc PR
2. implementation plan PR
3. implementation PR (TDD task-by-task)
4. release tag after green CI and required checks
