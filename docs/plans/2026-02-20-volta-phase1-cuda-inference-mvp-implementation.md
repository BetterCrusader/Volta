# Volta Phase 1 CUDA Inference MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship CUDA inference MVP through the same verified IR and execution plan path used by CPU, with deterministic strict mode and parity validation.

**Architecture:** Introduce a runtime gateway that routes execution by backend capabilities while preserving a single truth path (`verify_graph -> build_execution_plan -> runtime gateway -> backend run`). CUDA support is inference-only in this phase and must not add alternate executors.

**Tech Stack:** Rust (`src/ir`, `src/model`), Rust integration tests (`tests/*.rs`), shell CI scripts (`scripts/ci/*.sh`), GitHub Actions workflows.

---

## Branching Strategy

1. Create milestone branch from `main`:
   `git checkout -b milestone/cuda-inference-mvp-v0.3`
2. Keep commits task-scoped (one task per commit).
3. No direct pushes to `main`; merge through PR with required checks.

## CI Strategy for Phase 1

- PR required checks: fmt, clippy, unit/integration compile, policy checks.
- GPU hardware checks (parity/perf) run in nightly lanes, not required on every PR.
- `scripts/ci/cuda_infer_verify.sh` is required in local verification and can be run in CI lanes where CUDA is available.

## Task 1: Backend capability contract

**Files:**
- Create: `tests/backend_capabilities.rs`
- Create: `src/ir/backend_capabilities.rs`
- Modify: `src/ir/backend.rs`
- Modify: `src/ir/mod.rs`

**Step 1: Write the failing test**

Add test asserting CPU and CUDA capability reporting includes deterministic support flags and supported kernel classes.

**Step 2: Run test to verify RED**

Run: `cargo test backend_capabilities -- --nocapture`
Expected: FAIL with missing type/function symbols.

**Step 3: Write minimal implementation**

Add `BackendCapabilities` and `DeterminismLevel` types and expose `capabilities()` on `Backend` trait.

**Step 4: Re-run test for GREEN**

Run: `cargo test backend_capabilities -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "feat(ir): add backend capability contract"`

## Task 2: Runtime gateway (single truth path)

**Files:**
- Create: `src/ir/runtime.rs`
- Create: `tests/runtime_single_truth_path.rs`
- Modify: `src/ir/mod.rs`
- Modify: `src/model/train_api.rs`
- Modify: `src/ir/train.rs`

**Step 1: Write the failing test**

Test asserts infer and train execution call runtime gateway APIs rather than backend-specific direct execution.

**Step 2: Run RED**

Run: `cargo test runtime_single_truth_path -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Add runtime gateway functions that accept `ExecutionPlan`, backend selector, and `ExecutionContext`.

**Step 4: Run GREEN**

Run: `cargo test runtime_single_truth_path -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "refactor(runtime): route infer/train through runtime gateway"`

## Task 3: Shape binding for memory byte accounting

**Files:**
- Create: `tests/memory_shape_binding.rs`
- Modify: `src/model/builder.rs`
- Modify: `src/ir/shape_inference.rs`
- Modify: `src/ir/memory_planner.rs`

**Step 1: Write failing test**

Test asserts tiny-transformer plan reports non-zero `peak_live_bytes` when input/parameter shapes are bound.

**Step 2: Run RED**

Run: `cargo test memory_shape_binding -- --nocapture`
Expected: FAIL with zero-byte estimate.

**Step 3: Minimal implementation**

Introduce shape bindings for `Input`/`Parameter` in builder/model metadata and consume them during shape inference/planning.

**Step 4: Run GREEN**

Run: `cargo test memory_shape_binding tiny_transformer_peak_memory_stays_within_budget -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "feat(memory): bind input and parameter shapes for byte accounting"`

## Task 4: Execution plan backend placement hints

**Files:**
- Create: `tests/execution_plan_backend_hints.rs`
- Modify: `src/ir/execution_plan.rs`
- Modify: `src/ir/allocation.rs`

**Step 1: Write failing test**

Assert execution plan includes backend-neutral placement metadata for values/buffers.

**Step 2: Run RED**

Run: `cargo test execution_plan_backend_hints -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Add placement hint structs to plan and fill from allocation/storage classes.

**Step 4: Run GREEN**

Run: `cargo test execution_plan_backend_hints -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "feat(plan): add backend placement hints to execution plan"`

## Task 5: CUDA module scaffold and compile path

**Files:**
- Create: `src/ir/cuda/mod.rs`
- Create: `src/ir/cuda/device.rs`
- Create: `src/ir/cuda/lowering.rs`
- Create: `src/ir/cuda/executor.rs`
- Create: `tests/cuda_backend_scaffold.rs`
- Modify: `src/ir/mod.rs`
- Modify: `src/ir/backend.rs`

**Step 1: Write failing scaffold test**

Assert CUDA backend compiles plan object and reports explicit unsupported op errors for missing kernels.

**Step 2: Run RED**

Run: `cargo test cuda_backend_scaffold -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Add scaffold types and no-op executor that fails explicitly.

**Step 4: Run GREEN**

Run: `cargo test cuda_backend_scaffold -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "feat(cuda): add backend scaffold and lowering placeholders"`

## Task 6: Kernel boundary dispatcher

**Files:**
- Create: `src/ir/cuda/kernels/mod.rs`
- Create: `src/ir/cuda/kernels/matmul.rs`
- Create: `src/ir/cuda/kernels/add.rs`
- Create: `src/ir/cuda/kernels/relu.rs`
- Create: `src/ir/cuda/kernels/softmax.rs`
- Create: `tests/cuda_kernel_dispatch.rs`
- Modify: `src/ir/cuda/executor.rs`
- Modify: `src/ir/kernel_grouping.rs`

**Step 1: Write failing dispatch test**

Assert kernel groups map to supported CUDA kernel handlers for inference op subset.

**Step 2: Run RED**

Run: `cargo test cuda_kernel_dispatch -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Add dispatch table and kernel stubs for supported op groups.

**Step 4: Run GREEN**

Run: `cargo test cuda_kernel_dispatch -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "feat(cuda): add inference kernel dispatch boundary"`

## Task 7: Strict deterministic CUDA mode

**Files:**
- Create: `src/ir/cuda/determinism.rs`
- Create: `tests/cuda_infer_determinism.rs`
- Modify: `src/ir/compiler_flags.rs`
- Modify: `src/ir/backend.rs`
- Modify: `src/ir/runtime.rs`

**Step 1: Write failing determinism test**

Run same CUDA inference twice in strict mode and assert exact output parity under defined tolerance policy.

**Step 2: Run RED**

Run: `cargo test cuda_infer_determinism -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Add strict determinism policy and fail-fast for unsupported nondeterministic kernels.

**Step 4: Run GREEN**

Run: `cargo test cuda_infer_determinism -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "feat(cuda): enforce strict deterministic inference mode"`

## Task 8: CPU/CUDA parity harness

**Files:**
- Create: `tests/cuda_infer_parity.rs`
- Create: `benchmarks/baselines/cuda-infer-parity.json`

**Step 1: Write failing parity test**

Assert CPU/CUDA inference parity (`max_abs_diff <= 1e-6`) on tiny-transformer fixture.

**Step 2: Run RED**

Run: `cargo test cuda_infer_parity -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation updates**

Wire parity harness through runtime gateway and backend selector.

**Step 4: Run GREEN**

Run: `cargo test cuda_infer_parity -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "test(cuda): add cpu-cuda parity gate for inference"`

## Task 9: CUDA memory guard

**Files:**
- Create: `tests/cuda_infer_memory_guard.rs`
- Create: `benchmarks/baselines/cuda-infer-memory-peak-bytes.txt`

**Step 1: Write failing memory guard test**

Test fails if baseline is missing or CUDA memory exceeds baseline + tolerance.

**Step 2: Run RED**

Run: `cargo test cuda_infer_memory_guard -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Add guard and baseline load path.

**Step 4: Run GREEN**

Run: `cargo test cuda_infer_memory_guard -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "test(cuda): add memory regression guard with baseline"`

## Task 10: CI lane and local verify script

**Files:**
- Create: `scripts/ci/cuda_infer_verify.sh`
- Modify: `.github/workflows/pr-gates.yml`
- Modify: `.github/workflows/nightly-quality.yml`

**Step 1: Write failing CI/script references**

Add references to script/job first.

**Step 2: Run RED**

Run: `bash scripts/ci/cuda_infer_verify.sh`
Expected: FAIL before script exists.

**Step 3: Minimal implementation**

Create script and wire PR/nightly jobs according to CI strategy.

**Step 4: Run GREEN**

Run: `bash scripts/ci/cuda_infer_verify.sh`
Expected: PASS.

**Step 5: Commit**

`git commit -m "ci: add cuda inference verify script and workflow lanes"`

## Task 11: Docs and governance sync

**Files:**
- Create: `docs/governance/cuda-determinism-policy.md`
- Modify: `README.md`
- Modify: `tests/docs_sync.rs`

**Step 1: Write failing docs sync assertions**

Add assertions for CUDA inference milestone and deterministic policy text.

**Step 2: Run RED**

Run: `cargo test --test docs_sync -- --nocapture`
Expected: FAIL.

**Step 3: Minimal implementation**

Update docs and policy sections.

**Step 4: Run GREEN**

Run: `cargo test --test docs_sync -- --nocapture`
Expected: PASS.

**Step 5: Commit**

`git commit -m "docs: sync cuda inference deterministic policy and README"`

## Task 12: Phase close verification and PR prep

**Files:**
- No new files expected unless fixups are required.

**Step 1: Run full local verification**

Run:
- `bash scripts/ci/wave1_local_verify.sh`
- `bash scripts/ci/wave23_local_verify.sh`
- `bash scripts/ci/tiny_transformer_cpu_verify.sh`
- `bash scripts/ci/cuda_infer_verify.sh`
- `cargo test --no-run`

Expected: all PASS.

**Step 2: If any failures appear, fix with TDD and re-run full verification.**

**Step 3: Commit verification fixups (if needed)**

`git commit -m "chore: phase1 cuda inference verification fixups"`

**Step 4: Push milestone branch**

`git push -u origin milestone/cuda-inference-mvp-v0.3`

**Step 5: Open PR to main with full test evidence**

Use `gh pr create` and include strict determinism and parity evidence in test plan.
