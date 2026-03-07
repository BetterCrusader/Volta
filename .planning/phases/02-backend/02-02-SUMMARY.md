---
phase: 02-backend
plan: 02
subsystem: codegen-portability
tags: [portability, include_bytes, gemm_shim, merged_rustflags, tests]
dependency_graph:
  requires: []
  provides: [portable-gemm-shim-embed, merged-rustflags-test-coverage]
  affects: [mlp_train_codegen, inner, mlp_train_rust_codegen]
tech_stack:
  added: []
  patterns: [include_bytes! for compile-time C file embedding, write-to-runtime-dir pattern]
key_files:
  created: []
  modified:
    - src/engine/ir/codegen/mlp_train_codegen.rs
    - src/engine/ir/codegen/inner.rs
    - src/engine/ir/codegen/mlp_train_rust_codegen.rs
decisions:
  - "include_bytes! used for gemm_shim.c — embed at compile time, write to per-compile dir at runtime; no CARGO_MANIFEST_DIR needed"
  - "merged_rustflags tests added alongside existing 3 tests — 5 new required names + #[allow(unsafe_code)] on tests mod"
metrics:
  duration: "6 minutes"
  completed: "2026-03-07T06:43:06Z"
  tasks_completed: 2
  files_modified: 3
requirements_closed: [INFRA-02, PERF-03]
---

# Phase 02 Plan 02: Portable gemm_shim + merged_rustflags Tests Summary

**One-liner:** Replaced CARGO_MANIFEST_DIR path embedding with include_bytes! at both gemm_shim.c sites; added 5 unit tests locking in merged_rustflags() behavior.

---

## What Was Built

### Task 1: Replace CARGO_MANIFEST_DIR with include_bytes! at both gemm_shim.c sites

Both `mlp_train_codegen.rs` and `inner.rs` previously contained:
```rust
let shim_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engine/ir/codegen/gemm_shim.c");
```

This embedded the dev machine's absolute path (`C:/Users/User/...`) into the shipped binary, breaking gemm_shim compilation on any install target.

Fix applied to both sites:
- Added `const GEMM_SHIM_C: &[u8] = include_bytes!("gemm_shim.c");` (compile-time embed)
- Replaced runtime path lookup with `std::fs::write(&shim_src, GEMM_SHIM_C)` into the per-compile working directory
- Added `#[cfg(test)]` sanity checks: `gemm_shim_bytes_not_empty` and `shim_bytes_not_empty`

`mlp_train_codegen.rs` writes to `out_dll.with_extension("shim.c")`.
`inner.rs` writes to `obj.with_file_name("gemm_shim.c")`.

### Task 2: Add unit tests for merged_rustflags() (PERF-03)

Added 5 required named tests to `mlp_train_rust_codegen.rs`:
- `merged_rustflags_injects_native_when_empty` — `merged_rustflags(None) == "-C target-cpu=native"`
- `merged_rustflags_injects_native_when_blank` — `merged_rustflags(Some("  ")) == "-C target-cpu=native"`
- `merged_rustflags_no_duplicate_when_already_set` — no double injection when native already present
- `merged_rustflags_no_duplicate_when_different_cpu` — leaves `target-cpu=haswell` unchanged
- `merged_rustflags_appends_when_other_flags_present` — appends native when other flags exist

---

## Verification Results

```
cargo test --lib -- merged_rustflags
running 8 tests
........
test result: ok. 8 passed; 0 failed

cargo test --lib
test result: ok. 221 passed; 0 failed; 2 ignored

cargo build --release
Finished `release` profile [optimized] target(s) in 29.83s

grep CARGO_MANIFEST_DIR mlp_train_codegen.rs inner.rs
(no output — clean)

strings volta.exe | grep "Users/User"
(no output — no dev paths in binary)
```

---

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pre-existing unsafe test compile errors in mlp_train_rust_codegen.rs tests**
- **Found during:** Task 2 (running `cargo test --lib merged_rustflags`)
- **Issue:** `std::env::remove_var` / `set_var` in test functions triggered `#![deny(unsafe_code)]` from `lib.rs`. Tests had been written with `unsafe {}` blocks, but linter had been stripping them, causing compile failures.
- **Fix:** Added `#[allow(unsafe_code)]` on the `mod tests` block; wrapped `remove_var`/`set_var` calls in `unsafe {}` blocks with SAFETY comments.
- **Files modified:** `src/engine/ir/codegen/mlp_train_rust_codegen.rs`
- **Commit:** 68f7ffb (included in Task 2 commit)

**2. [Rule 3 - Blocking] Linter repeatedly changed `resolve_mkl_lib_path` import in tests**
- **Found during:** Task 2
- **Issue:** Auto-formatter changed `use super::{merged_rustflags, resolve_mkl_lib_path}` to `resolve_mkl_lib_path_from`, breaking compilation since tests still called `resolve_mkl_lib_path()`.
- **Fix:** Restored import to `resolve_mkl_lib_path` for the tests that called it. The linter subsequently refactored those tests to use `resolve_mkl_lib_path_from` directly (no env mutation), which is actually cleaner.
- **Files modified:** `src/engine/ir/codegen/mlp_train_rust_codegen.rs`

---

## Commits

| Hash | Message |
|------|---------|
| 67786f5 | feat(02-02): replace CARGO_MANIFEST_DIR with include_bytes! at both gemm_shim.c sites |
| 68f7ffb | feat(02-02): add unit tests for merged_rustflags() and fix pre-existing unsafe test compile errors |

---

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| mlp_train_codegen.rs exists | FOUND |
| inner.rs exists | FOUND |
| mlp_train_rust_codegen.rs exists | FOUND |
| Commit 67786f5 exists | FOUND |
| Commit 68f7ffb exists | FOUND |
| GEMM_SHIM_C in mlp_train_codegen.rs | FOUND |
| GEMM_SHIM_C in inner.rs | FOUND |
| merged_rustflags_injects_native_when_empty test | FOUND |
