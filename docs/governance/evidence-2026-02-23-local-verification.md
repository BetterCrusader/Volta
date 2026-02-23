# Local Verification Evidence (2026-02-23)

This record captures local quality-gate verification run in the working copy on 2026-02-23.

## Environment

- Host: Windows (local workstation)
- Project path: `C:\Users\User\Desktop\Volta`
- Repository metadata: no `.git` directory in this working copy

## Commands and outcomes

### Core quality gates

- `cargo fmt --check` -> pass
- `cargo clippy --all-targets --all-features -- -D warnings` -> pass
- `cargo test --all-targets --all-features` -> pass
- `cargo test --features onnx-import` -> pass
- `cargo check --manifest-path fuzz/Cargo.toml` -> pass

### CI lane parity scripts

- `bash scripts/ci/wave1_local_verify.sh` -> pass
- `bash scripts/ci/wave23_local_verify.sh` -> pass
- `bash scripts/ci/cuda_infer_verify.sh` -> pass
- `bash scripts/ci/cuda_train_verify.sh` -> pass
- `bash scripts/ci/xl_verify.sh` -> pass
- `bash scripts/ci/interop_onnx_verify.sh` -> pass

### Determinism and governance spot checks

- `cargo test --test determinism_regression` -> pass (`3 passed`)
- `cargo test --test pass_equivalence` -> pass (`11 passed`)
- `cargo test --test governance_docs_content` -> pass (`6 passed`)
- `cargo test --test governance_docs_presence` -> pass (`1 passed`)

### CI stale reference sweep

- stale `scripts.ci.tests.*` references -> not found
- stale `schedule_optimization` references -> not found
- evidence note: `docs/governance/ci-stale-reference-sweep-2026-02-23.md`

### Release/performance gates

- `bash scripts/ci/release_perf_double_pass.sh` -> pass
  - pass 1: passed, no perf regression over threshold
  - pass 2: passed, no perf regression over threshold

### ONNX Wave 2.5 progress verification

- `cargo test --features onnx-import --test interop_onnx_wave2_parser` -> pass (`12 passed`, includes LeakyRelu default/custom alpha)
- `bash scripts/ci/interop_onnx_verify.sh` -> pass (contract + acceptance suites)
- `cargo test --features onnx-import` -> pass (full suite)

### Rollback verification

- `bash scripts/release/rollback.sh --verify-only` -> blocked in this working copy
  - reason: `fatal: not a git repository (or any of the parent directories): .git`
  - note: rollback verification requires repository tag history and must be run in a real git clone

## Result summary

- All executable local gates available in this workspace passed.
- The only remaining gate (`rollback --verify-only`) is environment-blocked because this workspace does not include git history.
