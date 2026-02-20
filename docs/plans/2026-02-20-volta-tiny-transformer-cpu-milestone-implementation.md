# Volta Tiny-Transformer CPU Milestone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a deterministic CPU-only tiny-transformer milestone that can train, infer, save, reload, and infer again without PyTorch.

**Architecture:** Keep one runtime truth path: model builder -> verified IR -> execution plan -> train/infer runtime -> checkpoint round-trip. Enforce correctness first with TDD and deterministic gates, then stabilize memory/planner/perf behavior.

**Tech Stack:** Rust (`src/model`, `src/ir`), integration tests (`tests/*.rs`), shell CI scripts (`scripts/ci/*.sh`), existing governance/quality workflows.

---

### Task 1: Add acceptance test skeleton for CPU milestone

**Files:**
- Create: `tests/tiny_transformer_cpu_e2e.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn tiny_transformer_cpu_train_infer_save_load_roundtrip() {
    let (model, dataset, train_config, infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    let trained = volta::model::train(&model, &dataset, &train_config).expect("train");
    let _before = volta::model::infer(&model, &trained.final_parameters, &infer_input).expect("infer");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture`
Expected: FAIL with missing symbols (`build_tiny_transformer_fixture_for_tests`, `infer`)

**Step 3: Add minimal compile stubs**

```rust
pub fn infer(...) -> Result<Tensor, TrainApiError> { unimplemented!() }
```

**Step 4: Run test to verify failure moves to runtime/logic**

Run: `cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture`
Expected: FAIL at runtime (`unimplemented!`) but compile succeeds

**Step 5: Commit**

```bash
git add tests/tiny_transformer_cpu_e2e.rs
git commit -m "test: add tiny-transformer CPU e2e acceptance skeleton"
```

### Task 2: Implement tiny-transformer fixture/model builder API

**Files:**
- Create: `src/model/tiny_transformer.rs`
- Modify: `src/model/mod.rs`
- Test: `src/model/tiny_transformer.rs`

**Step 1: Write the failing unit test**

```rust
#[test]
fn tiny_transformer_fixture_builds_verified_graph() {
    let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
    volta::ir::verify_graph(&model.graph).expect("graph must verify");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test tiny_transformer_fixture_builds_verified_graph -- --nocapture`
Expected: FAIL until fixture function and module export exist

**Step 3: Write minimal implementation**

```rust
pub fn build_tiny_transformer_fixture_for_tests() -> (CompiledModel, TinyDataset, TrainApiConfig, std::collections::HashMap<String, Tensor>) {
    // fixed-shape tiny transformer-like block using existing ops (matmul/add/relu/softmax)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test tiny_transformer_fixture_builds_verified_graph -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/tiny_transformer.rs src/model/mod.rs
git commit -m "feat: add tiny-transformer fixture/model builder API"
```

### Task 3: Add explicit inference API on runtime path

**Files:**
- Modify: `src/model/train_api.rs`
- Modify: `src/model/mod.rs`
- Test: `src/model/train_api.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn infer_returns_output_tensor_with_expected_shape() {
    let (model, _dataset, _cfg, infer_input) = crate::model::build_tiny_transformer_fixture_for_tests();
    let out = crate::model::infer(&model, &model.parameters, &infer_input).expect("infer");
    assert_eq!(out.shape, model.output_shape.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test infer_returns_output_tensor_with_expected_shape -- --nocapture`
Expected: FAIL until `infer` is implemented

**Step 3: Write minimal implementation**

```rust
pub fn infer(
    model: &CompiledModel,
    parameters: &std::collections::HashMap<String, Tensor>,
    inputs: &std::collections::HashMap<String, Tensor>,
) -> Result<Tensor, TrainApiError> {
    // build ExecutionContext, execute model.output via schedule, convert RuntimeValue -> Tensor
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test infer_returns_output_tensor_with_expected_shape -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/train_api.rs src/model/mod.rs
git commit -m "feat: add model inference API on verified runtime path"
```

### Task 4: Make CPU E2E acceptance test green (train + infer)

**Files:**
- Modify: `tests/tiny_transformer_cpu_e2e.rs`
- Modify: `src/model/tiny_transformer.rs`

**Step 1: Extend failing assertions**

```rust
assert!(trained.final_loss < baseline_loss);
assert_eq!(before.shape, after.shape);
```

**Step 2: Run test to verify it fails with current fixture/config**

Run: `cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture`
Expected: FAIL on loss/parity assertions

**Step 3: Adjust minimal fixture/config to satisfy acceptance**

```rust
let train_config = TrainApiConfig { epochs: 40, batch_size: 2, shuffle: true, shuffle_seed: 11, ... };
```

**Step 4: Re-run acceptance test**

Run: `cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/tiny_transformer_cpu_e2e.rs src/model/tiny_transformer.rs
git commit -m "feat: pass tiny-transformer CPU train/infer acceptance gate"
```

### Task 5: Introduce checkpoint format v1 header and metadata

**Files:**
- Modify: `src/model/checkpoint.rs`
- Test: `src/model/checkpoint.rs`

**Step 1: Write failing tests for v1 header + backward compatibility**

```rust
#[test]
fn checkpoint_v1_roundtrip_preserves_parameters() { /* ... */ }

#[test]
fn checkpoint_loader_accepts_legacy_format() { /* ... */ }
```

**Step 2: Run failing tests**

Run: `cargo test checkpoint_v1_roundtrip_preserves_parameters checkpoint_loader_accepts_legacy_format -- --nocapture`
Expected: FAIL before parser/writer update

**Step 3: Minimal implementation**

```rust
// write prefix lines:
// #volta-checkpoint:v1
// #created_by:train_api
```

**Step 4: Re-run tests**

Run: `cargo test roundtrip_checkpoint_is_deterministic checkpoint_v1_roundtrip_preserves_parameters checkpoint_loader_accepts_legacy_format -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/checkpoint.rs
git commit -m "feat: add checkpoint format v1 with legacy-compatible loader"
```

### Task 6: Add infer parity test across save/load round-trip

**Files:**
- Modify: `tests/tiny_transformer_cpu_e2e.rs`

**Step 1: Write failing parity assertion**

```rust
assert!(max_abs_diff(&before.data, &after.data) <= 1e-6);
```

**Step 2: Run test to verify failure**

Run: `cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture`
Expected: FAIL until checkpoint + infer wiring is complete

**Step 3: Minimal fix in test wiring/config**

```rust
let train_config = TrainApiConfig { checkpoint_path: Some(path.clone()), ..train_config };
```

**Step 4: Re-run test**

Run: `cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/tiny_transformer_cpu_e2e.rs
git commit -m "test: enforce infer parity after checkpoint reload"
```

### Task 7: Add memory planner peak-budget guard for tiny-transformer graph

**Files:**
- Create: `tests/tiny_transformer_memory_budget.rs`

**Step 1: Write failing memory budget test**

```rust
#[test]
fn tiny_transformer_peak_memory_stays_within_budget() {
    let (model, _, _, _) = volta::model::build_tiny_transformer_fixture_for_tests();
    let plan = volta::ir::plan_memory(&model.graph).expect("plan");
    assert!(plan.peak_live_bytes <= 2_000_000);
}
```

**Step 2: Run failing test**

Run: `cargo test tiny_transformer_peak_memory_stays_within_budget -- --nocapture`
Expected: FAIL if current graph/planner exceeds budget

**Step 3: Minimal implementation adjustment**

```rust
// adjust fixture dims/constants to fixed tiny budget-friendly values
```

**Step 4: Re-run test**

Run: `cargo test tiny_transformer_peak_memory_stays_within_budget -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/tiny_transformer_memory_budget.rs src/model/tiny_transformer.rs
git commit -m "test: add tiny-transformer memory planner peak budget guard"
```

### Task 8: Add deterministic replay test for tiny-transformer training

**Files:**
- Create: `tests/tiny_transformer_determinism.rs`

**Step 1: Write failing replay test**

```rust
#[test]
fn tiny_transformer_training_is_repeatable_for_fixed_seed() {
    // run train twice with same config/seed; compare final_loss and selected params
}
```

**Step 2: Run failing test**

Run: `cargo test tiny_transformer_training_is_repeatable_for_fixed_seed -- --nocapture`
Expected: FAIL until tolerance/seed wiring is correct

**Step 3: Minimal fix for deterministic config path**

```rust
let cfg = TrainApiConfig { reproducibility: ReproducibilityMode::Deterministic, shuffle_seed: 19, ..cfg };
```

**Step 4: Re-run test**

Run: `cargo test tiny_transformer_training_is_repeatable_for_fixed_seed -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/tiny_transformer_determinism.rs
git commit -m "test: add deterministic replay gate for tiny-transformer training"
```

### Task 9: Add local verification script for milestone gates

**Files:**
- Create: `scripts/ci/tiny_transformer_cpu_verify.sh`
- Modify: `README.md`

**Step 1: Write script-first failing run note in README**

```bash
bash scripts/ci/tiny_transformer_cpu_verify.sh
```

**Step 2: Run command to verify failure (script missing)**

Run: `bash scripts/ci/tiny_transformer_cpu_verify.sh`
Expected: FAIL with file not found

**Step 3: Minimal script implementation**

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo test tiny_transformer_cpu_train_infer_save_load_roundtrip -- --nocapture
cargo test tiny_transformer_training_is_repeatable_for_fixed_seed -- --nocapture
cargo test tiny_transformer_peak_memory_stays_within_budget -- --nocapture
```

**Step 4: Re-run script**

Run: `bash scripts/ci/tiny_transformer_cpu_verify.sh`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/ci/tiny_transformer_cpu_verify.sh README.md
git commit -m "chore: add local verify script for tiny-transformer CPU milestone"
```

### Task 10: Sync docs and guard tests with new milestone contract

**Files:**
- Modify: `tests/docs_sync.rs`
- Modify: `README.md`
- Modify: `docs/plans/2026-02-20-volta-engine-runtime-roadmap-design.md`

**Step 1: Add failing docs sync assertion**

```rust
assert!(text.contains("tiny-transformer CPU milestone"));
```

**Step 2: Run docs sync test to verify failure**

Run: `cargo test --test docs_sync -- --nocapture`
Expected: FAIL before README/design wording sync

**Step 3: Minimal docs update**

```text
Add explicit milestone section + verify command in README.
```

**Step 4: Re-run docs sync test**

Run: `cargo test --test docs_sync -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/docs_sync.rs README.md docs/plans/2026-02-20-volta-engine-runtime-roadmap-design.md
git commit -m "docs: sync tiny-transformer CPU milestone contract and verification"
```

### Task 11: Full verification sweep before milestone close

**Files:**
- No new files (verification only)

**Step 1: Run Wave 1 baseline**

Run: `bash scripts/ci/wave1_local_verify.sh`
Expected: PASS

**Step 2: Run Wave 2/3 baseline**

Run: `bash scripts/ci/wave23_local_verify.sh`
Expected: PASS

**Step 3: Run tiny-transformer verify script**

Run: `bash scripts/ci/tiny_transformer_cpu_verify.sh`
Expected: PASS

**Step 4: Run targeted release tests for touched modules**

Run: `cargo test --release model::train_api::tests::deterministic_mode_produces_stable_result model::checkpoint::tests::roundtrip_checkpoint_is_deterministic -- --nocapture`
Expected: PASS

**Step 5: Commit verification snapshot (if docs/report files changed)**

```bash
git add -A
git commit -m "chore: record verification-complete state for CPU milestone"
```

### Task 12: Post-milestone handoff for CUDA inference MVP (non-blocking)

**Files:**
- Create: `docs/plans/2026-02-20-volta-cuda-inference-mvp-design.md`

**Step 1: Write failing requirement checklist (doc TODOs)**

```markdown
- [ ] CPU parity harness defined
- [ ] Kernel boundary API frozen
- [ ] Inference-only operator subset selected
```

**Step 2: Validate current repo lacks CUDA MVP design doc**

Run: `test -f docs/plans/2026-02-20-volta-cuda-inference-mvp-design.md`
Expected: FAIL (missing file)

**Step 3: Write minimal design handoff doc**

```markdown
CPU-locked contracts reused; CUDA starts with inference parity before perf.
```

**Step 4: Re-check doc existence**

Run: `test -f docs/plans/2026-02-20-volta-cuda-inference-mvp-design.md`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/plans/2026-02-20-volta-cuda-inference-mvp-design.md
git commit -m "docs: add CUDA inference MVP handoff after CPU milestone"
```

## Notes for Execution

- Use `@superpowers/test-driven-development` discipline per task.
- Before any success claim or PR, use `@superpowers/verification-before-completion`.
- Keep commits small and task-aligned; avoid combining unrelated changes.
- Do not expand operator scope beyond milestone needs unless a blocking test requires it.
